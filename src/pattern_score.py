"""How "colourful repeating pattern" are the dense decoded vectors
(dmt_dense.py), and does that follow the condition, the drug level along
the recording, or the questionnaire?

Score of a vector: mean cosine to text descriptions of colourful repeating
geometric patterns minus mean cosine to descriptions of plain single-subject
photos (OpenCLIP ViT-H/14 text tower, the space the decoder writes into).

  conditions   whole-recording score, DMT vs eyes closed / open, same person
  kinetics     within each DMT recording, Spearman between the frame score
               (20 s kernel) and the measured drug level (alpha suppression,
               drug_curve.py) and the muscle proxy
  scales       DMT score, and DMT minus eyes closed, vs questionnaire

Output: outputs/pattern_score.json
"""
import json

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, wilcoxon

from common import DATA, OUTPUTS
from drug_curve import BIN
from dmt_events_video import STEP, trajectory

PATTERN = ["a colorful repeating geometric pattern", "a kaleidoscopic pattern", "a mandala", "a fractal pattern",
           "an ornate colorful tessellation", "a psychedelic pattern of bright colors", "many small colorful shapes",
           "a dense mosaic of saturated colors"]
PLAIN = ["a photo of an animal in nature", "a photo of a person standing", "a photo of a single object on a table",
         "a photo of a room", "a photo of a landscape", "a photo of a tool", "a photo of a building",
         "a plain beige surface"]


def axis():
    name = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model, _, _ = open_clip.create_model_and_transforms(name)
    tok = open_clip.get_tokenizer(name)
    with torch.no_grad():
        a, b = (F.normalize(model.eval().encode_text(tok(t)).float(), dim=-1) for t in (PATTERN, PLAIN))
    return a, b


def main():
    a, b = axis()
    score = lambda v: (F.normalize(v, dim=-1) @ a.T).mean(-1) - (F.normalize(v, dim=-1) @ b.T).mean(-1)
    d = torch.load(DATA / "processed/dmt_dense.pt", weights_only=False)
    curves = np.load(DATA / "processed/drug_curve.npz")
    sc = pd.read_csv(DATA / "raw/dmt/scales_results.csv", index_col=0, decimal=",")
    sc.index = [f"S{int(''.join(filter(str.isdigit, str(s)))):02d}" for s in sc.index]
    sc = sc.apply(pd.to_numeric, errors="coerce")
    rows, kin = [], []
    for key, rec in d.items():
        _, subj, cond = key.split("_")
        t, v = rec["t"], rec["v"].float()
        rows.append({"s": subj, "c": cond, "score": float(score(v.mean(0)))})
        if cond == "DMT" and f"{subj}_EC" in curves:
            clock = torch.arange(float(t.min()), float(t.max()) + STEP, STEP)
            s_t = score(trajectory(t, v, clock, 20.0)).numpy()
            tb = (np.arange(len(curves[f"{subj}_DMT"])) + 0.5) * BIN
            lvl = lambda col: np.interp(clock.numpy(), tb, np.log(curves[f"{subj}_DMT"][:, col] / np.median(curves[f"{subj}_EC"][:, col])))
            kin.append({"s": subj, "rho_drug": float(spearmanr(s_t, -lvl(0)).correlation),
                        "rho_muscle": float(spearmanr(s_t, lvl(2)).correlation),
                        "rho_time": float(spearmanr(s_t, clock.numpy()).correlation)})
    df = pd.DataFrame(rows).pivot(index="s", columns="c", values="score")
    kin = pd.DataFrame(kin).set_index("s")
    rep = {"mean_score": {c: float(df[c].mean()) for c in df}, "conditions": {}, "kinetics": {}, "scales": {}}
    for other in ("EC", "EO"):
        both = df.dropna(subset=["DMT", other])
        rep["conditions"][f"DMT_minus_{other}"] = {"n": len(both), "mean": float((both.DMT - both[other]).mean()),
                                                   "DMT_higher_in": int((both.DMT > both[other]).sum()),
                                                   "p": float(wilcoxon(both.DMT - both[other]).pvalue)}
    for k in ("rho_drug", "rho_muscle", "rho_time"):
        rep["kinetics"][k] = {"n": len(kin), "median": float(kin[k].median()), "positive": int((kin[k] > 0).sum()),
                              "p": float(wilcoxon(kin[k]).pvalue)}
    for name in ("5D-Total", "5D-ElementaryImagery", "5D-ComplexImagery"):
        for label, col in (("DMT", df.DMT), ("DMT_minus_EC", df.DMT - df.EC)):
            ok = col.dropna()
            q = spearmanr(ok, sc[name].reindex(ok.index))
            rep["scales"][f"{label}_vs_{name}"] = {"n": len(ok), "rho": float(q.correlation), "p": float(q.pvalue)}
    rep["per_subject"] = {s: {"DMT": None if pd.isna(r.DMT) else round(float(r.DMT), 4),
                              "EC": None if pd.isna(r.EC) else round(float(r.EC), 4),
                              "rho_drug": None if s not in kin.index else round(float(kin.loc[s, "rho_drug"]), 2)}
                          for s, r in df.iterrows()}
    (OUTPUTS / "pattern_score.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("mean_score", "conditions", "kinetics", "scales")}, indent=1))
    print({s: rep["per_subject"][s] for s in ("S28", "S04", "S29", "S35", "S12") if s in rep["per_subject"]})


if __name__ == "__main__":
    main()
