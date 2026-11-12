"""Gate for the direct decoder: zero-shot identification of which of 10 images
a person in another lab is SEEING (imagery dataset, perception phase).

Usage: python src/gate_perception.py encoder_sub-08_vi.pt [more checkpoints]

Each checkpoint is evaluated with and without per-recording whitening.
Subjects are split in two halves (odd/even position) so a configuration can
be chosen on one half and confirmed on the other.
"""
import json
import sys

import numpy as np
import torch
from scipy.stats import wilcoxon

from align import offdiag_corr, whiten_by_group
from analyze_imagery import IMAGES, identification, stimulus_embeddings
from common import DATA, OUTPUTS
from model_variants import build, run


def evaluate(emb, subj, image, sel_phase, stim):
    rows = {}
    for s in sorted(set(subj)):
        means = []
        for k in IMAGES:
            sel = sel_phase & (subj == s) & (image == k)
            means.append(emb[torch.from_numpy(sel)].mean(0) if sel.any() else None)
        if any(v is None for v in means):
            continue
        means = torch.stack(means)
        rows[s] = identification(means - means.mean(0), stim)
    return rows


def summarize(rows, subjects):
    r = np.array([rows[s]["mean_rank"] for s in subjects if s in rows])
    return {"n": len(r), "acc10": float(np.mean([rows[s]["acc10"] for s in subjects if s in rows])),
            "within": float(np.mean([rows[s]["acc_within"] for s in subjects if s in rows])),
            "mean_rank": float(r.mean()), "p": float(wilcoxon(r - 5.5, alternative="less").pvalue)}


def main():
    w = np.load(DATA / "processed/imagery_windows.npz")
    x, subj, ses, task, image, phase = (w[k] for k in ("x", "subject", "session", "task", "image", "phase"))
    rec = np.char.add(np.char.add(subj, ses), task)
    print("mean |off-diagonal channel correlation| before whitening:", round(offdiag_corr(x[::20]), 3), flush=True)
    xw = whiten_by_group(x, rec)
    print("after whitening:", round(offdiag_corr(xw[::20]), 3), flush=True)
    stim = stimulus_embeddings()
    subjects = sorted(set(subj))
    halves = {"A": subjects[0::2], "B": subjects[1::2], "all": subjects}
    report = {}
    for ckpt in sys.argv[1:]:
        for name, data in (("raw", x), ("whitened", xw)):
            m = build(ckpt, "original", data[phase == "rest"])
            emb = run(m, data)
            for ph in ("perception", "imagery"):
                rows = evaluate(emb, subj, image, phase == ph, stim)
                res = {h: summarize(rows, ss) for h, ss in halves.items()}
                report[f"{ckpt}|{name}|{ph}"] = res
                print(ckpt, name, ph, json.dumps({h: {k: round(v, 3) for k, v in r.items()} for h, r in res.items()}),
                      flush=True)
    out = OUTPUTS / "gate_perception_report.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(report)
    out.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
