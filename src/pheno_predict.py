"""Predict visual phenomenology scores from the drug-induced change in EEG
band power.

Train on PsiConnect (psilocybin minus sober baseline; 11D-ASC elementary
imagery, complex imagery, audio-visual synaesthesia), choose the feature set
by leave-one-subject-out inside PsiConnect only, then test once on the
Zenodo DMT dataset (DMT minus eyes closed; 5D-ASC same three subscales).
Features are z-scored within each dataset, so only the ranking of subjects
is compared (Spearman), with a label-permutation p-value.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, OUTPUTS, SHARED_CHANNELS
from render import load_scales

REGIONS = {"frontal": ["Fp1", "Fp2", "F7", "F8", "FC1", "FC2"],
           "central": ["Cz", "C3", "C4", "T7", "T8", "CPz", "CP1", "CP2", "CP5", "CP6"],
           "posterior": ["Pz", "P3", "P4", "O1", "O2"]}
TARGETS = {"elementary": ("ASC11_ELEMENTARY", "5D-ElementaryImagery"),
           "complex": ("ASC11_COMPLEX", "5D-ComplexImagery"),
           "audiovisual": ("ASC11_AUDIOVISUAL", "5D-AudioVisualSyn")}
PSI_TASKS = ("rest", "meditation", "music")
ALPHAS = np.logspace(-1, 4, 20)


def feature_sets(d):
    """d: (n, 21, 5) log-power change."""
    reg = np.stack([d[:, [SHARED_CHANNELS.index(c) for c in chs]].mean(1) for chs in REGIONS.values()], 1)
    return {"full": d.reshape(len(d), -1), "regional": reg.reshape(len(d), -1), "global": d.mean(1)}


def load():
    w = np.load(DATA / "processed/pheno_features.npz")
    psi = pd.DataFrame({"s": w["psi_subject"], "ses": w["psi_session"], "task": w["psi_task"],
                        "i": np.arange(len(w["psi_subject"]))})
    psi = psi[psi.task.isin(PSI_TASKS)]
    xs, ids = [], []
    for s, g in psi.groupby("s"):
        a, b = g[g.ses == "ses-02"], g[g.ses == "ses-01"]
        common = sorted(set(a.task) & set(b.task))
        if not common:
            continue
        xs.append(np.mean([w["psi_feat"][a[a.task == t].i.iloc[0]] - w["psi_feat"][b[b.task == t].i.iloc[0]]
                           for t in common], 0))
        ids.append(s)
    asc = pd.read_csv(DATA / "raw/psiconnect/phenotype/ASC11.tsv", sep="\t", index_col=0).reindex(ids)
    psi_x, psi_y = np.stack(xs), asc

    dm = pd.DataFrame({"s": w["dmt_subject"], "c": w["dmt_condition"], "i": np.arange(len(w["dmt_subject"]))})
    xs, ids = [], []
    for s, g in dm.groupby("s"):
        if {"DMT", "EC"} <= set(g.c):
            xs.append(w["dmt_feat"][g[g.c == "DMT"].i.iloc[0]] - w["dmt_feat"][g[g.c == "EC"].i.iloc[0]])
            ids.append(s)
    return psi_x, psi_y, np.stack(xs), load_scales().reindex(ids), ids


def model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def perm_p(pred, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearmanr(pred, y).statistic
    null = np.array([spearmanr(pred, rng.permutation(y)).statistic for _ in range(n)])
    return float(obs), float((np.sum(null >= obs) + 1) / (n + 1))


def main():
    psi_x, psi_y, dmt_x, dmt_y, dmt_ids = load()
    fs_psi, fs_dmt = feature_sets(psi_x), feature_sets(dmt_x)
    report = {"n_psi": len(psi_x), "n_dmt": len(dmt_x)}
    preds = {"subject": dmt_ids}
    for name, (col_psi, col_dmt) in TARGETS.items():
        y = psi_y[col_psi].to_numpy(float)
        ok = ~np.isnan(y)
        loso = {}
        for fs, X in fs_psi.items():
            p = cross_val_predict(model(), X[ok], y[ok], cv=LeaveOneOut())
            loso[fs] = perm_p(p, y[ok])
        best = max(loso, key=lambda k: loso[k][0])
        m = model().fit(fs_psi[best][ok], y[ok])
        Xd = fs_dmt[best]
        Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-9) * fs_psi[best][ok].std(0) + fs_psi[best][ok].mean(0)
        pred = m.predict(Xd)
        yd = dmt_y[col_dmt].to_numpy(float)
        okd = ~np.isnan(yd)
        rho, p = perm_p(pred[okd], yd[okd])
        pw = cross_val_predict(model(), fs_dmt[best][okd], yd[okd], cv=LeaveOneOut())
        report[name] = {"psi_loso": {k: {"rho": v[0], "p": v[1]} for k, v in loso.items()},
                        "chosen_features": best,
                        "cross_dataset_dmt": {"rho": rho, "p": p},
                        "dmt_within_loso": dict(zip(("rho", "p"), perm_p(pw, yd[okd])))}
        preds[f"{name}_pred"] = pred.tolist()
        preds[f"{name}_reported"] = yd.tolist()
    (OUTPUTS / "pheno_report.json").write_text(json.dumps(report, indent=1))
    (OUTPUTS / "pheno_predictions.json").write_text(json.dumps(preds, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
