"""Do the decoded vectors follow the reported stages of an intense DMT
experience (geometric patterns at the beginning and end, complex
architecture and entities at the peak)?

Text probes per category (OpenCLIP ViT-H/14 text tower); a vector's score
for a category is its mean cosine to that category's probes minus its mean
over all categories. Within each DMT recording, Spearman between the frame
score (20 s kernel) and the measured drug level (alpha suppression relative
to the person's eyes closed, drug_curve.py), for three trajectories:
  real        dense template-aligned windows (dmt_dense.py)
  surrogate   the same pipeline on the phase-randomized copy, which keeps
              the spectrum and its slow time course: what the spectrum
              alone predicts
  local       real minus surrogate at the same moment

Prediction from the reports: geometry falls with the drug level, entities
and architecture rise. Output: outputs/stage_kinetics.json
"""
import json

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, wilcoxon

from common import DATA, OUTPUTS
from dmt_events_video import STEP, trajectory
from drug_curve import BIN
from test_text_probes import PROBES

EXTRA = {"architecture": ["impossible architecture", "a vast ornate palace interior", "a colorful temple with intricate walls",
                          "a futuristic alien city", "a hyperdimensional structure", "a cathedral made of light",
                          "a room with complex ornate walls", "a giant intricate machine", "a dome covered in patterns",
                          "a labyrinth of corridors"]}


def probes():
    name = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model, _, _ = open_clip.create_model_and_transforms(name)
    tok = open_clip.get_tokenizer(name)
    cats = {**PROBES, **EXTRA}
    with torch.no_grad():
        return {k: F.normalize(model.eval().encode_text(tok(v)).float(), dim=-1) for k, v in cats.items()}


def main():
    pr = probes()
    names = list(pr)

    def scores(v):
        s = torch.stack([(F.normalize(v, dim=-1) @ pr[k].T).mean(-1) for k in names], -1)
        return s - s.mean(-1, keepdim=True)

    d = torch.load(DATA / "processed/dmt_dense.pt", weights_only=False)
    curves = np.load(DATA / "processed/drug_curve.npz")
    rows = []
    for key, rec in d.items():
        _, subj, cond = key.split("_")
        if cond != "DMT" or f"{subj}_EC" not in curves or len(rec["ts"]) < 100:
            continue
        t, v, ts, vs = rec["t"], rec["v"].float(), rec["ts"], rec["vs"].float()
        clock = torch.arange(float(t.min()), float(t.max()) + STEP, STEP)
        real, sur = trajectory(t, v, clock, 20.0), trajectory(ts, vs, clock, 20.0)
        tb = (np.arange(len(curves[f"{subj}_DMT"])) + 0.5) * BIN
        lvl = -np.interp(clock.numpy(), tb, np.log(curves[f"{subj}_DMT"][:, 0] / np.median(curves[f"{subj}_EC"][:, 0])))
        for kind, tr in (("real", real), ("surrogate", sur), ("local", real - sur)):
            sc = scores(tr).numpy()
            for j, k in enumerate(names):
                rows.append({"s": subj, "kind": kind, "cat": k, "rho": float(spearmanr(sc[:, j], lvl).correlation),
                             "first2": float(sc[clock.numpy() < 120, j].mean()) if (clock < 120).any() else np.nan,
                             "later": float(sc[clock.numpy() >= 180, j].mean()) if (clock >= 180).any() else np.nan})
    df = pd.DataFrame(rows)
    rep = {}
    for (kind, cat), g in df.groupby(["kind", "cat"]):
        el = g.dropna(subset=["first2", "later"])
        rep.setdefault(kind, {})[cat] = {"n": len(g), "median_rho": round(float(g.rho.median()), 3),
                                         "positive": int((g.rho > 0).sum()), "p": round(float(wilcoxon(g.rho).pvalue), 4),
                                         "first2min_minus_later": round(float((el.first2 - el.later).mean()), 4),
                                         "p_early_late": round(float(wilcoxon(el.first2 - el.later).pvalue), 4)}
    (OUTPUTS / "stage_kinetics.json").write_text(json.dumps(rep, indent=1))
    t = pd.DataFrame({(kind, m): {c: v[m] for c, v in cats.items()} for kind, cats in rep.items()
                      for m in ("median_rho", "positive", "p", "p_early_late")})
    print(t[["real", "surrogate", "local"]].to_string())


if __name__ == "__main__":
    main()
