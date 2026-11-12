"""Is there any DMT-specific signal in the decoded embeddings?

Raw per-subject means collapse onto each decoder's default output (identical to
Gaussian noise), so we analyse within-subject contrasts instead:
  c_s = mean(DMT windows) - mean(EC windows)
and ask (1) whether c_s points the same way across subjects beyond a
label-permutation null, (2) whether ATM and linear agree, (3) which THINGS
concepts the group contrast points to, (4) whether the per-subject projection
tracks the 5D imagery scales. EO-EC serves as a reference for a real effect.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from common import DATA, OUTPUTS
from embed_dmt import MODEL_NAMES, concept_names
from render import load_scales

N_PERM = 200


def subject_contrasts(e, subj, cond, subjects, a, b, rng=None):
    out = []
    for s in subjects:
        ia = np.where((subj == s) & (cond == a))[0]
        ib = np.where((subj == s) & (cond == b))[0]
        if rng is not None:
            pool = rng.permutation(np.concatenate([ia, ib]))
            ia, ib = pool[:len(ia)], pool[len(ia):]
        out.append(e[ia].mean(0) - e[ib].mean(0))
    return torch.stack(out)


def consistency(c):
    n = F.normalize(c, dim=-1)
    sim = n @ n.T
    return sim[~torch.eye(len(c), dtype=bool)].mean().item()


def main():
    d = torch.load(DATA / "processed/dmt_embeddings.pt", weights_only=False)
    subj, cond = d["subject"], d["condition"]
    scales = load_scales()
    rng = np.random.default_rng(0)

    def n(s, c):
        return int(((subj == s) & (cond == c)).sum())

    subjects = [s for s in sorted(set(subj)) if n(s, "DMT") >= 30 and n(s, "EC") >= 30]
    eo_subjects = [s for s in subjects if n(s, "EO") >= 30]

    feats = torch.load(DATA / "raw/things/ViT-H-14_features_train.pt")["img_features"].float()
    concepts = concept_names()
    cf = feats.reshape(len(concepts), 10, -1).mean(1)
    cf = F.normalize(cf - cf.mean(0), dim=-1)

    report, group = {"n_subjects": len(subjects), "n_subjects_eo": len(eo_subjects)}, {}
    for m in MODEL_NAMES:
        e = d["emb"][m]
        c = subject_contrasts(e, subj, cond, subjects, "DMT", "EC")
        c_eo = subject_contrasts(e, subj, cond, eo_subjects, "EO", "EC")
        null = [consistency(subject_contrasts(e, subj, cond, subjects, "DMT", "EC", rng))
                for _ in range(N_PERM)]
        obs = consistency(c)
        g = c.mean(0)
        group[m] = g
        proj = (c @ F.normalize(g, dim=0)).numpy()
        sc = scales.reindex(subjects)
        corr = {k: spearmanr(proj, sc[k], nan_policy="omit") for k in sc.columns}
        top = (cf @ F.normalize(g, dim=0)).topk(15)
        bottom = (-(cf @ F.normalize(g, dim=0))).topk(10)
        report[m] = {
            "dmt_ec_consistency": obs,
            "dmt_ec_null_mean": float(np.mean(null)),
            "dmt_ec_null_p": float((np.sum(np.array(null) >= obs) + 1) / (N_PERM + 1)),
            "eo_ec_consistency": consistency(c_eo),
            "cos_group_dmt_vs_group_eo": F.cosine_similarity(g, c_eo.mean(0), dim=0).item(),
            "more_with_dmt": [concepts[i] for i in top.indices],
            "less_with_dmt": [concepts[i] for i in bottom.indices],
            "spearman_projection_vs_scales": {k: {"rho": float(v.statistic), "p": float(v.pvalue)}
                                              for k, v in corr.items()},
        }
    report["cos_group_atm_vs_linear"] = F.cosine_similarity(group["atm"], group["linear"], dim=0).item()
    torch.save(group, DATA / "processed/group_contrasts.pt")
    (OUTPUTS / "contrast_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
