"""PsiConnect (psilocybin 19 mg vs sober baseline, 62-channel EEG).

1. Positive control: sober cloud movie vs sober rest. If the THINGS-trained
   decoder transfers to this lab at all, the contrast should rank "cloud" high.
2. Psilocybin effect per task: ses-02 minus ses-01, per subject.
3. Clouds under psilocybin vs sober: does the decoded "face" score track how
   many faces each person reported seeing in the clouds during the EEG?
"""
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

from common import DATA, OUTPUTS
from decode import ConceptSpace, embed, group_mean_null, load_psiconnect, sign_flip_null

TASKS = ("rest", "meditation", "music", "clouds")


def main():
    x, meta = load_psiconnect()
    emb = embed(x, "encoder_sub-08_psi.pt")
    subj, ses, task = meta["subject"], meta["session"], meta["task"]
    torch.save({"emb": emb, "subject": subj, "session": ses, "task": task, "t": meta["t"]},
               DATA / "processed/psiconnect_embeddings.pt")
    cs = ConceptSpace()
    ph = DATA / "raw/psiconnect/phenotype"
    clouds = pd.read_csv(ph / "CLOUDS.tsv", sep="\t", index_col=0)
    asc = pd.read_csv(ph / "ASC11.tsv", sep="\t", index_col=0)

    means = {}
    for key in pd.MultiIndex.from_arrays([subj, ses, task]).unique():
        sel = (subj == key[0]) & (ses == key[1]) & (task == key[2])
        if sel.sum() >= 30:
            means[key] = emb[torch.from_numpy(sel)].mean(0)
    subjects = sorted({k[0] for k in means})

    def contrast(pairs):
        ids, vecs = [], []
        for s in subjects:
            a, b = (means.get((s,) + p) for p in pairs)
            if a is not None and b is not None:
                ids.append(s)
                vecs.append(a - b)
        return ids, torch.stack(vecs)

    def summarize(ids, c, extra_concepts=()):
        obs, null_mean, p = sign_flip_null(c)
        _, p_norm = group_mean_null(c)
        g = c.mean(0)
        out = {"n_subjects": len(ids), "consistency": obs, "null_mean": null_mean,
               "p_consistency": p, "p_group_mean": p_norm, "top": cs.top(g), "bottom": cs.top(-g)}
        for k in extra_concepts:
            ranks = [cs.rank(v, k) for v in c]
            out[f"rank_{k}_group"] = cs.rank(g, k)
            out[f"rank_{k}_median_subject"] = float(np.median(ranks))
            out[f"rank_{k}_wilcoxon_p_vs_middle"] = float(
                wilcoxon(np.array(ranks) - (len(cs.names) + 1) / 2, alternative="less").pvalue)
        return out, g

    report, vectors = {"n_concepts": len(cs.names)}, {}
    report["control_sober_clouds_vs_rest"], vectors["sober_clouds_vs_rest"] = summarize(
        *contrast([("ses-01", "clouds"), ("ses-01", "rest")]), extra_concepts=("cloud",))
    report["control_psilo_clouds_vs_rest"], vectors["psilo_clouds_vs_rest"] = summarize(
        *contrast([("ses-02", "clouds"), ("ses-02", "rest")]), extra_concepts=("cloud",))
    for t in TASKS:
        report[f"psilocybin_{t}"], vectors[f"psilocybin_{t}"] = summarize(
            *contrast([("ses-02", t), ("ses-01", t)]), extra_concepts=("face", "kaleidoscope"))

    ids, c = contrast([("ses-02", "clouds"), ("ses-01", "clouds")])
    face = np.array([cs.scores(v)[cs.names.index("face")].item() for v in c])
    rep = clouds.reindex(ids)
    report["clouds_face_score_vs_reported_faces"] = {
        k: dict(zip(("rho", "p"), map(float, spearmanr(face, rep[k], nan_policy="omit"))))
        for k in ("CLOUDS_EEGFACES", "CLOUDS_EEGOBJECTS", "CLOUDS_EEGPATTERNS")}

    ids, c = contrast([("ses-02", "rest"), ("ses-01", "rest")])
    proj = (c @ torch.nn.functional.normalize(c.mean(0), dim=0)).numpy()
    a = asc.reindex(ids)
    report["rest_projection_vs_asc11"] = {
        k: dict(zip(("rho", "p"), map(float, spearmanr(proj, a[k], nan_policy="omit"))))
        for k in ("ASC11_COMPLEX", "ASC11_ELEMENTARY", "ASC11_AUDIOVISUAL")}

    torch.save(vectors, DATA / "processed/psiconnect_group_vectors.pt")
    (OUTPUTS / "psiconnect_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
