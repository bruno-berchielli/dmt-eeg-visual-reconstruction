"""Split-half reliability of the event-decoded trajectory, every recording.

For each recording with enough events, dmt_events_video.halves_report: do
two independent halves of the events change the same way over time? Then:
group level per condition, DMT vs eyes closed in the same person, and the
relation with the questionnaire scores and the amount of data.

Usage: python halves_group.py [T500multi]
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon

from common import DATA, OUTPUTS
from dmt_events_video import STEP, halves_report, load

MIN_EVENTS = 16


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "T500multi"
    d = torch.load(DATA / f"processed/dmt_events_{kind}.pt", weights_only=False)
    rows = []
    for v in d.values():
        if len(v["events_s"]) < MIN_EVENTS or v["surrogate_clip_mean"] is None:
            continue
        t, x = load(v["subject"], kind, v["condition"])
        r = halves_report(t, x, torch.arange(float(t.min()), float(t.max()) + STEP, STEP), n_perm=1000)
        rows.append({"s": v["subject"], "c": v["condition"], "n": len(t), "minutes": v["minutes"],
                     "same_time": r["same_time_cos"], "p": r["p"], "mean_cos": r["mean_cos"]})
    df = pd.DataFrame(rows)
    sc = pd.read_csv(DATA / "raw/dmt/scales_results.csv", index_col=0, decimal=",")
    sc.index = [f"S{int(''.join(filter(str.isdigit, str(s)))):02d}" for s in sc.index]
    sc = sc.apply(pd.to_numeric, errors="coerce")
    rep = {"detector": kind, "conditions": {}}
    for c, g in df.groupby("c"):
        rep["conditions"][c] = {"n": len(g), "median_same_time": float(g.same_time.median()),
                                "mean_same_time": float(g.same_time.mean()),
                                "p_group_gt0": float(wilcoxon(g.same_time, alternative="greater").pvalue),
                                "n_p_below_0.05": int((g.p < 0.05).sum())}
    piv = df.pivot(index="s", columns="c", values="same_time")
    both = piv.dropna(subset=["DMT", "EC"])
    rep["DMT_vs_EC"] = {"n": len(both), "mean_diff": float((both.DMT - both.EC).mean()),
                        "p": float(wilcoxon(both.DMT - both.EC).pvalue)}
    dmt = df[df.c == "DMT"].set_index("s")
    rep["DMT_same_time_vs"] = {}
    for name, col in (("5D-Total", sc["5D-Total"]), ("5D-ElementaryImagery", sc["5D-ElementaryImagery"]),
                      ("5D-ComplexImagery", sc["5D-ComplexImagery"]), ("n_events", dmt.n), ("minutes", dmt.minutes)):
        r = spearmanr(dmt.same_time, col.reindex(dmt.index))
        rep["DMT_same_time_vs"][name] = {"rho": float(r.correlation), "p": float(r.pvalue)}
    (OUTPUTS / f"halves_group_{kind}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    t = df.pivot(index="s", columns="c", values=["same_time", "p", "n"]).round(2)
    t["total5D"] = sc["5D-Total"].reindex(t.index)
    print(t.sort_values("total5D", ascending=False).to_string())


if __name__ == "__main__":
    main()
