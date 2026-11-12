"""Lean cross-lab check of the THINGS-defined onset detectors (matched-filter
templates and onset nets saved by onset_things.py): on the imagery dataset,
where screen changes happen at known times, is the detector score higher at
those times than elsewhere in the trial?

For each recording the score is z-scored over the whole recording, then
averaged across trials on a 20 ms grid from -8 to +9 s around the marker
(imagery onset). Known screen changes: image on at about -6.0 s, image off
/ mask at about -2.0 s, black screen at 0, rest cue at about +4.25 s.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

from common import DATA, MODELS, OUTPUTS, SFREQ
from onset_lib import CHANNEL_SETS, CONT, STEP, OnsetNet, clean_mask, load_cont, mf_score, net_score

TAU = np.arange(-8.0, 9.0001, STEP / SFREQ)


def score_recording(x, kind, cset, templates):
    if kind == "net":
        ck = torch.load(MODELS / f"onset_net_{cset}.pt", map_location="cpu")
        model = OnsetNet(len(ck["channels"]), ck["n_times"])
        model.load_state_dict(ck["state_dict"])
        starts = np.arange(0, x.shape[1] - ck["n_times"] + 1, STEP)
        sc = np.full(x.shape[1], np.nan, np.float32)
        v = np.repeat(net_score(model, x, starts, ck["n_times"], "cpu"), STEP)
        sc[:len(v)] = v[:len(sc)]
        return sc
    t = templates[f"{cset}_{kind}"]
    sc = np.full(x.shape[1], np.nan, np.float32)
    sc[:x.shape[1] - t.shape[1] + 1] = mf_score(x, t)
    return sc


def main():
    cset = sys.argv[1] if len(sys.argv) > 1 else "z14"
    templates = np.load(MODELS / "onset_templates.npz")
    kinds = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("T200", "T500", "net")
    acc = {k: [] for k in kinds}
    for path in sorted(CONT.glob("imagery_*.npz")):
        _, sub, ses, task = path.stem.split("_")
        x, pieces, _ = load_cont(path, CHANNEL_SETS[cset])
        good = clean_mask(x, pieces)
        ev = pd.read_csv(DATA / f"raw/imagery/extracted/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_events.tsv", sep="\t")
        for k in kinds:
            sc = score_recording(x, k, cset, templates)
            sc[~good] = np.nan
            sc = (sc - np.nanmean(sc)) / np.nanstd(sc)
            rows = []
            for o in ev.onset:
                idx = np.round((o + TAU) * SFREQ).astype(int)
                if idx[0] >= 0 and idx[-1] < len(sc):
                    rows.append(sc[idx])
            if rows:
                acc[k].append((sub, np.nanmean(rows, 0)))
        print(path.stem, flush=True)
    report = {}
    for k in kinds:
        subs = sorted({s for s, _ in acc[k]})
        per = np.stack([np.nanmean([c for s2, c in acc[k] if s2 == s], 0) for s in subs])
        mean, sem = per.mean(0), per.std(0) / np.sqrt(len(per))
        order = np.argsort(-mean)[:12]
        report[k] = {"n_subjects": len(subs), "tau": TAU.tolist(), "mean_z": mean.tolist(), "sem": sem.tolist(),
                     "top": [(round(float(TAU[i]), 2), round(float(mean[i]), 3), round(float(mean[i] / sem[i]), 1))
                             for i in sorted(order)]}
        print(k, "top (tau, z, t):", report[k]["top"])
    (OUTPUTS / f"detector_check_{cset}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
