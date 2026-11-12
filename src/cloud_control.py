"""Positive control and model selection on PsiConnect.

Sober participants (ses-01) watch a cloud movie vs rest. A decoder that
transfers to this lab should rank the THINGS concept "cloud" high in the
per-subject (clouds - rest) contrast. Each model variant is scored on this
alone; the winner is chosen before looking at any drug data.
"""
import json

import numpy as np
import torch
from scipy.stats import wilcoxon

from common import OUTPUTS
from decode import ConceptSpace, consistency, load_psiconnect
from model_variants import VARIANTS, build, run


def main():
    x, meta = load_psiconnect()
    subj, ses, task = meta["subject"], meta["session"], meta["task"]
    keep = (ses == "ses-01") & np.isin(task, ["rest", "clouds"])
    x = x[keep]
    subj, task = subj[keep], task[keep]
    baseline = x[task == "rest"]
    cs = ConceptSpace()
    n_c = len(cs.names)
    subjects = [s for s in sorted(set(subj))
                if ((subj == s) & (task == "rest")).sum() >= 30 and ((subj == s) & (task == "clouds")).sum() >= 30]

    report = {"n_subjects": len(subjects), "n_concepts": n_c}
    for v in VARIANTS:
        m = build("encoder_sub-08_psi.pt", v, baseline)
        e = run(m, x)
        c = torch.stack([e[torch.from_numpy((subj == s) & (task == "clouds"))].mean(0) -
                         e[torch.from_numpy((subj == s) & (task == "rest"))].mean(0) for s in subjects])
        ranks = np.array([cs.rank(vec, "cloud") for vec in c])
        report[v] = {
            "cloud_rank_group": cs.rank(c.mean(0), "cloud"),
            "cloud_rank_median_subject": float(np.median(ranks)),
            "cloud_top100_fraction": float((ranks <= 100).mean()),
            "wilcoxon_p_rank_below_middle": float(wilcoxon(ranks - (n_c + 1) / 2, alternative="less").pvalue),
            "consistency": consistency(c),
            "top": cs.top(c.mean(0), 12),
        }
        print(v, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in report[v].items()}), flush=True)
    best = min(VARIANTS, key=lambda v: (report[v]["cloud_rank_median_subject"], report[v]["cloud_rank_group"]))
    report["selected"] = best
    (OUTPUTS / "cloud_control_report.json").write_text(json.dumps(report, indent=1))
    print("selected:", best)


if __name__ == "__main__":
    main()
