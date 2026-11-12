"""Does aligned decoding transfer to another lab, and where exactly is the
image onset there?

A centered ridge decoder (60-500 ms after onset -> CLIP) is trained on
THINGS-EEG2 with the 30 channels shared with the imagery dataset. In that
dataset each trial shows one of 10 known images; its marker is the imagery
onset, the image appears about 6 s earlier. The decoder is applied at many
assumed onset times (lags) around every trial. For each lag: per subject,
mean decoded embedding per image, centered across the 10 images, matched to
the CLIP embeddings of the 10 pictures (zero-shot).

If decoding transfers, identification should rise only when the assumed
onset matches the true one, which also measures that latency. Lags where no
image appears are the built-in control.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from analyze_imagery import IMAGES, identification, stimulus_embeddings
from common import DATA, MODELS, OUTPUTS, SFREQ, THINGS_CHANNELS, WINDOW, normalize_windows
from onset_lib import CONT, VI_CHANNELS, adapt, clean_mask, load_cont
from train_encoder import load_features
from window_sweep import featurize

START, END = 60, 500
LAMBDA = 1e3
LAGS = np.round(np.r_[np.arange(-7.5, -6.5, 0.1), np.arange(-6.5, -5.3, 0.02), np.arange(-5.3, 3.01, 0.1)], 3)


def train_ridge():
    out = MODELS / "ridge_vi30_60-500ms.pt"
    if out.exists():
        return torch.load(out)
    idx = [THINGS_CHANNELS.index(c) for c in VI_CHANNELS]
    x = np.load(DATA / "processed/things_sub-08_training_63ch_f16.npy", mmap_mode="r")
    y = load_features("train").numpy().astype(np.float64)
    chunks = [np.arange(i, min(i + 1000, len(x))) for i in range(0, len(x), 1000)]
    feats = []
    for c in chunks:
        xc = normalize_windows(np.asarray(x[c][:, :, idx]).astype(np.float32))
        feats.append(featurize(xc, START, END).reshape(len(c) * x.shape[1], -1).astype(np.float32))
    z = np.concatenate(feats).astype(np.float64)
    mu, sd = z.mean(0), z.std(0) + 1e-6
    z = (z - mu) / sd
    yy = np.repeat(y, x.shape[1], axis=0)
    w = np.linalg.solve(z.T @ z + LAMBDA * np.eye(z.shape[1]), z.T @ (yy - yy.mean(0)))
    ck = {"w": torch.from_numpy(w).float(), "mu": torch.from_numpy(mu).float(), "sd": torch.from_numpy(sd).float()}
    torch.save(ck, out)
    return ck


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    ck = train_ridge()
    w, mu, sd = ck["w"].numpy(), ck["mu"].numpy(), ck["sd"].numpy()
    stim = stimulus_embeddings()
    sums, counts = {}, {}
    for path in sorted(CONT.glob("imagery_*.npz")):
        _, sub, ses, task = path.stem.split("_")
        x, pieces, d = load_cont(path, VI_CHANNELS)
        good = clean_mask(x, pieces)
        if mode:
            x, _ = adapt(x, good, mode)
        ev = pd.read_csv(DATA / f"raw/imagery/extracted/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_events.tsv",
                         sep="\t")
        for li, lag in enumerate(LAGS):
            starts = np.round((ev.onset.to_numpy() + lag) * SFREQ).astype(int)
            ok = (starts >= 0) & (starts + WINDOW <= x.shape[1])
            ok &= np.array([good[s:s + WINDOW].all() if o else False for s, o in zip(starts, ok)])
            if not ok.any():
                continue
            win = normalize_windows(np.stack([x[:, s:s + WINDOW] for s in starts[ok]]))
            emb = ((featurize(win, START, END) - mu) / sd) @ w
            for img, e in zip(ev["type"].to_numpy()[ok], emb):
                key = (sub, li, img)
                sums[key] = sums.get(key, 0) + e
                counts[key] = counts.get(key, 0) + 1
        print(path.stem, flush=True)

    subjects = sorted({k[0] for k in sums})
    curve = []
    for li, lag in enumerate(LAGS):
        ranks, within, acc = [], [], []
        for s in subjects:
            if not all((s, li, k) in sums for k in IMAGES):
                continue
            m = torch.tensor(np.stack([sums[(s, li, k)] / counts[(s, li, k)] for k in IMAGES])).float()
            r = identification(m - m.mean(0), stim)
            ranks.append(r["mean_rank"]); within.append(r["acc_within"]); acc.append(r["acc10"])
        ranks = np.array(ranks)
        curve.append({"lag": float(lag), "n": len(ranks), "mean_rank": float(ranks.mean()),
                      "acc10": float(np.mean(acc)), "within": float(np.mean(within)),
                      "p": float(wilcoxon(ranks - 5.5, alternative="less").pvalue)})
    tag = f"_{mode}" if mode else ""
    (OUTPUTS / f"lag_curve{tag}.json").write_text(json.dumps(curve, indent=1))
    best = sorted(curve, key=lambda c: c["mean_rank"])[:8]
    print("best lags:", json.dumps([{k: round(v, 4) for k, v in c.items()} for c in best]))
    ctrl = [c["mean_rank"] for c in curve if c["lag"] > -5.0 or c["lag"] < -6.6]
    print("control lags mean rank: %.3f (sd %.3f, min %.3f)" % (np.mean(ctrl), np.std(ctrl), np.min(ctrl)))


if __name__ == "__main__":
    main()
