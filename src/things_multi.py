"""Does training on several THINGS-EEG2 subjects help where one subject failed?

Jitter-tolerant centered ridge to CLIP (waveform, 20 ms bins, 100-540 ms,
copies shifted by -40..+40 ms), fitted from per-subject sufficient
statistics so any subset of subjects can be combined without reloading.

Measures, for the single-subject model (sub-08) and the multi-subject ones:
  1. THINGS, held-out person: train on the others, test on sub-08's test
     set (200-way, 80 repetitions averaged), aligned and jittered;
  2. other lab: zero-shot identification of the 10 imagery-dataset pictures
     during perception, at assumed onsets around the measured one
     (-5.814 s from the marker), 30 shared channels;
  3. saves the 21-channel multi-subject decoder for DMT and multi-subject
     matched-filter templates (14 channels).
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

import cache_things
from analyze_imagery import IMAGES, identification, stimulus_embeddings
from common import DATA, MODELS, OUTPUTS, SFREQ, SHARED_CHANNELS, THINGS_CHANNELS, WINDOW, normalize_windows
from loso_imagery import epochs_at
from onset_lib import CHANNEL_SETS, CONT, VI_CHANNELS, clean_mask, load_cont
from tolerance_test import A, B, bins, retrieval
from train_encoder import load_features

OFFS = (-10, -5, 0, 5, 10)
LAMBDA = 1e4
ONSET = -5.814
SETS = {"dmt21": SHARED_CHANNELS, "vi30": VI_CHANNELS}


def features(x, off=0):
    return bins(x, A + off, B + off, 5).reshape(len(x), -1)


def moments(sub, chans, y):
    out = MODELS / f"things_moments_sub-{sub}_{len(chans)}ch.npz"
    if out.exists():
        return dict(np.load(out))
    idx = [THINGS_CHANNELS.index(c) for c in chans]
    x = np.load(cache_things.cache_path("training", sub), mmap_mode="r")
    S = s = C = None
    n = 0
    for i in range(0, len(x), 1000):
        xc = normalize_windows(np.asarray(x[i:i + 1000][:, :, idx]).astype(np.float32).reshape(-1, len(idx), 250))
        yc = np.repeat(y[i:i + 1000], x.shape[1], axis=0)
        for o in OFFS:
            f = features(xc, o).astype(np.float64)
            S = f.T @ f if S is None else S + f.T @ f
            s = f.sum(0) if s is None else s + f.sum(0)
            C = f.T @ yc if C is None else C + f.T @ yc
            n += len(f)
    m = {"S": S, "s": s, "C": C, "n": np.array(n)}
    np.savez(out, **m)
    return m


def solve(ms, y_mean):
    S, s, C, n = (sum(m[k] for m in ms) for k in ("S", "s", "C", "n"))
    n = float(n)
    mu = s / n
    sd = np.sqrt(np.maximum(np.diag(S) / n - mu ** 2, 1e-12))
    zz = (S - n * np.outer(mu, mu)) / np.outer(sd, sd)
    zy = (C - n * np.outer(mu, y_mean)) / sd[:, None]
    return {"w": np.linalg.solve(zz + LAMBDA * np.eye(len(zz)), zy), "mu": mu, "sd": sd}


def predict(ck, x, off=0):
    return ((features(x, off) - ck["mu"]) / ck["sd"]) @ ck["w"]


def things_test(ck, sub, chans, f_test, rng):
    idx = [THINGS_CHANNELS.index(c) for c in chans]
    x = np.load(cache_things.cache_path("test", sub), mmap_mode="r")
    x = normalize_windows(np.asarray(x[:, :, idx]).astype(np.float32).reshape(-1, len(idx), 250))
    aligned = predict(ck, x).reshape(200, 80, -1).mean(1)
    jit = rng.integers(-10, 11, len(x))
    pj = np.empty((len(x), ck["w"].shape[1]), np.float32)
    for o in np.unique(jit):
        pj[jit == o] = predict(ck, x[jit == o], int(o))
    return {"aligned": retrieval(aligned, f_test), "jitter_pm40ms": retrieval(pj.reshape(200, 80, -1).mean(1), f_test)}


def imagery_gate(models, stim):
    lags = np.round(ONSET + np.arange(-0.2, 0.2001, 0.02), 3)
    sums = {name: {} for name in models}
    for path in sorted(CONT.glob("imagery_*.npz")):
        _, sub, ses, task = path.stem.split("_")
        x, pieces, _ = load_cont(path, VI_CHANNELS)
        good = clean_mask(x, pieces)
        ev = pd.read_csv(DATA / f"raw/imagery/extracted/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_events.tsv", sep="\t")
        for li, lag in enumerate(lags):
            ep, keep = epochs_at(x, good, ev.onset.to_numpy(), lag)
            if not len(ep):
                continue
            win = normalize_windows(ep)
            labels = ev["type"].to_numpy()[keep]
            for name, ck in models.items():
                emb = predict(ck, win)
                for img in IMAGES:
                    sel = labels == img
                    if sel.any():
                        a = sums[name].setdefault((sub, li, img), [0, 0])
                        a[0] = a[0] + emb[sel].sum(0)
                        a[1] += int(sel.sum())
    out = {}
    for name in models:
        curve = []
        subs = sorted({k[0] for k in sums[name]})
        for li, lag in enumerate(lags):
            ranks, within = [], []
            for s in subs:
                if all((s, li, k) in sums[name] for k in IMAGES):
                    m = torch.tensor(np.stack([sums[name][(s, li, k)][0] / sums[name][(s, li, k)][1] for k in IMAGES])).float()
                    r = identification(m - m.mean(0), stim)
                    ranks.append(r["mean_rank"]); within.append(r["acc_within"])
            ranks = np.array(ranks)
            curve.append({"lag": float(lag), "mean_rank": float(ranks.mean()), "within": float(np.mean(within)),
                          "p_better": float(wilcoxon(ranks - 5.5, alternative="less").pvalue)})
        at = min(curve, key=lambda c: abs(c["lag"] - ONSET))
        out[name] = {"at_measured_onset": at, "best": min(curve, key=lambda c: c["mean_rank"]),
                     "worst": max(curve, key=lambda c: c["mean_rank"]), "curve": curve}
    return out


def templates(subs):
    idx = [THINGS_CHANNELS.index(c) for c in CHANNEL_SETS["z14"]]
    acc = 0
    for sub in subs:
        x = np.load(cache_things.cache_path("training", sub), mmap_mode="r")
        pick = np.arange(0, len(x), 4)
        acc = acc + normalize_windows(np.asarray(x[pick][:, :, idx]).astype(np.float32).reshape(-1, len(idx), 250)).mean(0)
    t = acc / len(subs)
    np.savez(MODELS / "onset_templates_multi.npz", z14_T200=t[:, :50], z14_T500=t[:, :125])


def main():
    subs = sys.argv[1:] or ["08", "01", "02", "03", "05"]
    for sub in subs:
        cache_things.main(sub)
    y = load_features("train").numpy().astype(np.float64)
    y_mean = y.mean(0)
    f_test = load_features("test")
    rng = np.random.default_rng(0)
    report = {"subjects": subs}
    mom = {name: {sub: moments(sub, chans, y) for sub in subs} for name, chans in SETS.items()}
    others = [s for s in subs if s != "08"]

    cks = {"single_08": solve([mom["dmt21"]["08"]], y_mean), "all": solve([mom["dmt21"][s] for s in subs], y_mean)}
    if others:
        cks["others_only"] = solve([mom["dmt21"][s] for s in others], y_mean)
    report["things_test_sub08_21ch"] = {k: things_test(ck, "08", SHARED_CHANNELS, f_test, rng) for k, ck in cks.items()}
    print(json.dumps(report["things_test_sub08_21ch"], indent=1), flush=True)
    torch.save({"w": torch.from_numpy(cks["all"]["w"]).float(), "mu": torch.from_numpy(cks["all"]["mu"]).float(),
                "sd": torch.from_numpy(cks["all"]["sd"]).float(), "y_mean": torch.from_numpy(y_mean).float()},
               MODELS / "event_ridge_clip_multi.pt")

    gate_models = {"single_08": solve([mom["vi30"]["08"]], y_mean), "all": solve([mom["vi30"][s] for s in subs], y_mean)}
    gate = imagery_gate(gate_models, stimulus_embeddings())
    report["imagery_gate_30ch"] = {k: {a: v[a] for a in ("at_measured_onset", "best", "worst")} for k, v in gate.items()}
    print(json.dumps(report["imagery_gate_30ch"], indent=1), flush=True)
    (OUTPUTS / "things_multi_gate_curves.json").write_text(json.dumps({k: v["curve"] for k, v in gate.items()}))
    templates(subs)
    (OUTPUTS / "things_multi_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
