"""Is the cross-lab failure about people or about lab / protocol?

Inside the imagery dataset (22 people, same lab, same stimuli), leave one
subject out and build from the other 21:
  detector   matched-filter template = their average response in the first
             500 ms after the image appears (14 channels, as for DMT)
  decoder    ridge from the first 500 ms (20 ms bins, +-40 ms jittered
             copies) to a one-hot of the 10 images
and test on the held-out person. If this works, differences between people
are not the main obstacle and the THINGS -> other-lab gap comes from the lab
or the RSVP protocol; if it fails, people differ too much and more THINGS
subjects are the lever.

Image onset relative to the marker is taken as the lag that maximizes the
grand-average occipital response energy (estimated on all subjects; it is a
property of the paradigm, not of a person).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from analyze_imagery import IMAGES, TASK_OF
from common import DATA, OUTPUTS, SFREQ, WINDOW, normalize_windows
from onset_lib import CHANNEL_SETS, CONT, clean_mask, load_cont, mf_score
from tolerance_test import bins

L = 125
OFFS = (-10, -5, 0, 5, 10)


def epochs_at(x, good, onsets, lag, pre=0):
    out, keep = [], []
    for i, o in enumerate(onsets):
        s = int(round((o + lag) * SFREQ)) - pre
        if s >= 0 and s + WINDOW + pre <= x.shape[1] and good[s:s + WINDOW + pre].all():
            out.append(x[:, s:s + WINDOW + pre])
            keep.append(i)
    return (np.stack(out) if out else np.zeros((0, x.shape[0], WINDOW + pre), np.float32)), np.array(keep, int)


def find_onset(recs):
    """Grand-average posterior response around the nominal -6 s; onset = the
    time where its 100 ms energy first exceeds half of the peak."""
    acc, n = 0, 0
    post = [CHANNEL_SETS["z14"].index(c) for c in ("P3", "Pz", "P4", "O1", "O2")]
    for x, good, ev in recs:
        ep, _ = epochs_at(x, good, ev.onset.to_numpy(), -6.6)
        if len(ep):
            e = ep[:, post] - ep[:, post, :50].mean(-1, keepdims=True)
            acc, n = acc + e.sum(0), n + len(e)
    erp = acc / n
    energy = np.convolve((erp ** 2).mean(0), np.ones(25) / 25, mode="same")
    base = energy[:60].mean()
    first = np.argmax(energy > base + 0.5 * (energy.max() - base))
    return -6.6 + first / SFREQ - 0.05, erp


def main():
    chans = CHANNEL_SETS["z14"]
    by_sub = {}
    for path in sorted(CONT.glob("imagery_*.npz")):
        _, sub, ses, task = path.stem.split("_")
        x, pieces, _ = load_cont(path, chans)
        ev = pd.read_csv(DATA / f"raw/imagery/extracted/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_events.tsv", sep="\t")
        by_sub.setdefault(sub, []).append((x, clean_mask(x, pieces), ev))
    onset, _ = find_onset([r for rs in by_sub.values() for r in rs])
    print("estimated image onset relative to marker: %.3f s" % onset, flush=True)

    data = {}
    for sub, recs in by_sub.items():
        ep, lab = [], []
        for x, good, ev in recs:
            e, keep = epochs_at(x, good, ev.onset.to_numpy(), onset, pre=10)
            ep.append(e)
            lab += list(ev["type"].to_numpy()[keep])
        data[sub] = (np.concatenate(ep), np.array(lab))

    def feats(e, off):
        w = normalize_windows(e[:, :, 10 + off:10 + off + WINDOW])
        return bins(w, 0, L, 5).reshape(len(w), -1)

    subs = sorted(data)
    res = {"onset_s": onset, "detector_auc": {}, "decoder_acc10": {}, "decoder_within": {}}
    for held in subs:
        train = [s for s in subs if s != held]
        tpl = np.mean([normalize_windows(data[s][0][:, :, 10:10 + WINDOW])[:, :, :L].mean(0) for s in train], 0)
        pos, neg = [], []
        for x, good, ev in by_sub[held]:
            sc = np.full(x.shape[1], np.nan)
            sc[:x.shape[1] - L + 1] = mf_score(x, tpl)
            sc[~good] = np.nan
            for o in ev.onset:
                i = int(round((o + onset) * SFREQ))
                if 0 <= i < len(sc):
                    pos.append(np.nanmax(sc[max(0, i - 5):i + 6]))
                    for lag in (-1.5, 6.5, 7.5):
                        j = i + int(lag * SFREQ)
                        if 0 <= j < len(sc) - 6:
                            neg.append(np.nanmax(sc[max(0, j - 5):j + 6]))
        pos, neg = np.array(pos), np.array(neg)
        pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        res["detector_auc"][held] = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))

        Z = np.concatenate([feats(data[s][0], o) for s in train for o in OFFS]).astype(np.float64)
        Y = np.concatenate([np.eye(10)[[IMAGES.index(k) for k in data[s][1]]] for s in train for _ in OFFS])
        mu, sd = Z.mean(0), Z.std(0) + 1e-6
        Z = (Z - mu) / sd
        w = np.linalg.solve(Z.T @ Z + 1e3 * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
        zt = (feats(data[held][0], 0) - mu) / sd
        pred = zt @ w
        yt = np.array([IMAGES.index(k) for k in data[held][1]])
        res["decoder_acc10"][held] = float((pred.argmax(1) == yt).mean())
        within = []
        for p, t in zip(pred, yt):
            same = [j for j, k in enumerate(IMAGES) if TASK_OF[k] == TASK_OF[IMAGES[t]]]
            within.append(same[int(np.argmax(p[same]))] == t)
        res["decoder_within"][held] = float(np.mean(within))
        print(held, "detector AUC %.3f | decoder 10-way %.3f | within-category %.3f" % (
            res["detector_auc"][held], res["decoder_acc10"][held], res["decoder_within"][held]), flush=True)

    auc = np.array(list(res["detector_auc"].values()))
    acc = np.array(list(res["decoder_acc10"].values()))
    wit = np.array(list(res["decoder_within"].values()))
    res["summary"] = {"detector_auc_mean": float(auc.mean()), "detector_p": float(wilcoxon(auc - 0.5, alternative="greater").pvalue),
                      "decoder_acc10_mean": float(acc.mean()), "decoder_p": float(wilcoxon(acc - 0.1, alternative="greater").pvalue),
                      "decoder_within_mean": float(wit.mean()), "within_chance": 0.3}
    print(json.dumps(res["summary"], indent=1))
    (OUTPUTS / "loso_imagery_report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
