"""Detector and decoder trained on the imagery dataset's PERCEPTION phase
(one image at a time, 4 s on screen, 22 people), using the 14 channels shared
with the Zenodo DMT montage. loso_imagery.py showed this transfers to unseen
people inside that lab; here it is fitted on all 22 and checked across labs
in the reverse direction, on THINGS-EEG2 (sub-08), where onsets and images
are known:
  detection  matched-filter score at the true onset vs 100 ms off
  content    for THINGS concepts that match the 10 imagery images (dog, fish,
             bird, cup, chair, watch, scissors), does the decoder pick the
             right one among those present?

Saves models/vi_model.pt: template (14 x 125), ridge to one-hot of the 10
images (first 500 ms, 20 ms bins, +-40 ms jitter), image onset latency.
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from analyze_imagery import IMAGES
from common import DATA, MODELS, OUTPUTS, SFREQ, THINGS_CHANNELS, WINDOW, normalize_windows
from embed_dmt import concept_names
from loso_imagery import L, OFFS, epochs_at, find_onset
from onset_lib import CHANNEL_SETS, CONT, clean_mask, load_cont
from tolerance_test import bins

CHANS = CHANNEL_SETS["z14"]
THINGS_MATCH = {"dog": "dog", "fish": "fish", "bird": "bird", "cup": "cup", "chair": "chair",
                "watch": "watch", "scissor": "scissors"}


def feats(win):
    """win: (n, 14, 250) normalized windows starting at the assumed onset."""
    return bins(win, 0, L, 5).reshape(len(win), -1)


class VIModel:
    def __init__(self):
        ck = torch.load(MODELS / "vi_model.pt", weights_only=False)
        self.template, self.w, self.mu, self.sd, self.onset = (ck[k] for k in ("template", "w", "mu", "sd", "onset"))

    def scores(self, win):
        """Centered class scores, columns in analyze_imagery.IMAGES order."""
        return ((feats(win) - self.mu) / self.sd) @ self.w


def fit():
    recs = []
    for path in sorted(CONT.glob("imagery_*.npz")):
        _, sub, ses, task = path.stem.split("_")
        x, pieces, _ = load_cont(path, CHANS)
        ev = pd.read_csv(DATA / f"raw/imagery/extracted/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_events.tsv", sep="\t")
        recs.append((x, clean_mask(x, pieces), ev))
    onset, _ = find_onset(recs)
    ep, lab = [], []
    for x, good, ev in recs:
        e, keep = epochs_at(x, good, ev.onset.to_numpy(), onset, pre=10)
        ep.append(e)
        lab += list(ev["type"].to_numpy()[keep])
    ep = np.concatenate(ep)
    y = np.eye(10)[[IMAGES.index(k) for k in lab]]
    template = normalize_windows(ep[:, :, 10:10 + WINDOW])[:, :, :L].mean(0)
    Z = np.concatenate([feats(normalize_windows(ep[:, :, 10 + o:10 + o + WINDOW])) for o in OFFS]).astype(np.float64)
    Y = np.tile(y, (len(OFFS), 1))
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Z = (Z - mu) / sd
    w = np.linalg.solve(Z.T @ Z + 1e3 * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
    torch.save({"template": template.astype(np.float32), "w": w.astype(np.float32), "mu": mu.astype(np.float32),
                "sd": sd.astype(np.float32), "onset": onset, "n_trials": len(ep)}, MODELS / "vi_model.pt")
    return len(ep)


def check_on_things():
    m = VIModel()
    idx = [THINGS_CHANNELS.index(c) for c in CHANS]
    x = np.load(DATA / "processed/things_sub-08_training_63ch_f16.npy", mmap_mode="r")
    t = m.template - m.template.mean(1, keepdims=True)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(x), 1500, replace=False))
    ep = np.asarray(x[pick][:, :, idx]).astype(np.float32).reshape(-1, len(idx), 250)
    ep = ep - ep.mean(1, keepdims=True)

    def ncc(seg):
        s = seg - seg.mean(-1, keepdims=True)
        s = s / (s.std(-1, keepdims=True) + 1e-6)
        return (s * t).sum((1, 2)) / (np.linalg.norm(t) * np.sqrt(t.size))

    pos, neg = ncc(ep[:, :, :L]), ncc(ep[:, :, 25:25 + L])
    auc = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))

    concepts = concept_names()
    cols = [IMAGES.index(k) for k in THINGS_MATCH]
    hits_single, hits_img, n_img = [], [], 0
    for j, (vi_name, th_name) in enumerate(THINGS_MATCH.items()):
        if th_name not in concepts:
            continue
        c = concepts.index(th_name)
        e = np.asarray(x[c * 10:(c + 1) * 10][:, :, idx]).astype(np.float32)
        sc = m.scores(normalize_windows(e.reshape(-1, len(idx), 250)))[:, cols]
        hits_single += list(sc.argmax(1) == j)
        hits_img += list(sc.reshape(10, 4, -1).mean(1).argmax(1) == j)
        n_img += 1
    return {"detection_auc_onset_vs_100ms_off": auc, "matched_concepts": n_img,
            "content_single_trial_acc": float(np.mean(hits_single)),
            "content_per_image_acc": float(np.mean(hits_img)), "content_chance": 1 / max(n_img, 1)}


def main():
    n = fit()
    rep = {"n_training_trials": n, "onset_s": VIModel().onset, "things_check": check_on_things()}
    (OUTPUTS / "vi_model_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
