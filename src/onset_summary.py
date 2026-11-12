"""Part B statistics from outputs/onset_counts_<detector>.json: events per
minute and mean detector score per condition, real minus surrogate, paired
tests across subjects (Zenodo DMT / EC / EO; OSF drug x time), and the
imagery-lab positive control (real screen changes must beat surrogates).

Output: outputs/onset_partB_report.json
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from common import OUTPUTS

MIN_CLEAN_MIN = 1.0
METRICS = ["rate", "surr_rate", "rate_minus_surr", "mean_score", "score_minus_surr"]


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    return {"n": len(d), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
            "median_diff": float(np.median(d)), "n_a_greater": int((d > 0).sum()),
            "p_wilcoxon": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
            "p_ttest": float(ttest_rel(a, b).pvalue)}


def rows(name):
    out = json.loads((OUTPUTS / f"onset_counts_{name}.json").read_text())
    for r in out:
        if "rate" in r:
            r["score_minus_surr"] = r["mean_score"] - r["surr_mean_score"]
    return out


def imagery(rs):
    by = defaultdict(lambda: defaultdict(list))
    for r in rs:
        if r["recording"].startswith("imagery_") and r.get("clean_minutes", 0) >= MIN_CLEAN_MIN:
            sub = r["recording"].split("_")[1]
            for m in METRICS + ["mean_score", "surr_mean_score"]:
                by[sub][m].append(r[m])
    subs = sorted(by)
    real = [np.mean(by[s]["rate"]) for s in subs]
    surr = [np.mean(by[s]["surr_rate"]) for s in subs]
    sc = [np.mean(by[s]["mean_score"]) for s in subs]
    ssc = [np.mean(by[s]["surr_mean_score"]) for s in subs]
    return {"events_per_min_real_vs_surrogate": paired(real, surr), "mean_score_real_vs_surrogate": paired(sc, ssc)}


def zenodo(rs):
    d = defaultdict(dict)
    usable = defaultdict(list)
    for r in rs:
        if not r["recording"].startswith("dmt_"):
            continue
        _, s, c = r["recording"].split("_")
        usable[c].append((r.get("clean_minutes", 0.0), r["total_minutes"]))
        if r.get("clean_minutes", 0) >= MIN_CLEAN_MIN:
            d[s][c] = r
    out = {"usable_data": {c: {"recordings": len(v), "with_>=1_clean_min": int(sum(a >= MIN_CLEAN_MIN for a, _ in v)),
                               "clean_minutes_total": float(sum(a for a, _ in v)),
                               "recorded_minutes_total": float(sum(b for _, b in v)),
                               "clean_minutes_median": float(np.median([a for a, _ in v]))}
                           for c, v in usable.items()}}
    per_cond = {}
    for c in ("DMT", "EC", "EO"):
        v = [d[s][c] for s in d if c in d[s]]
        per_cond[c] = {m: {"mean": float(np.mean([r[m] for r in v])), "median": float(np.median([r[m] for r in v]))}
                       for m in METRICS}
        per_cond[c]["n"] = len(v)
    out["per_condition"] = per_cond
    for a, b in (("DMT", "EC"), ("EO", "EC"), ("DMT", "EO")):
        subs = sorted(s for s in d if a in d[s] and b in d[s])
        out[f"{a}_vs_{b}"] = {m: paired([d[s][a][m] for s in subs], [d[s][b][m] for s in subs]) for m in METRICS}
        out[f"{a}_vs_{b}"]["subjects"] = subs
    out["per_subject"] = {s: {c: {m: round(r[m], 3) for m in METRICS + ["clean_minutes"]} for c, r in v.items()}
                          for s, v in sorted(d.items())}
    return out


def osf(rs):
    d = defaultdict(dict)
    for r in rs:
        if not r["recording"].startswith("osf_") or r.get("clean_minutes", 0) < MIN_CLEAN_MIN:
            continue
        _, drug, s, seg = r["recording"].split("_")
        d[s][(drug, int(seg[-1]))] = r
    out = {"per_cell_mean": {}}
    for drug in ("DMT", "PLA"):
        for k in range(4):
            v = [d[s][(drug, k)] for s in d if (drug, k) in d[s]]
            out["per_cell_mean"][f"{drug}_seg{k}"] = {m: float(np.mean([r[m] for r in v])) for m in METRICS}
            out["per_cell_mean"][f"{drug}_seg{k}"]["n"] = len(v)
    for k in (1, 2, 3):
        subs = sorted(s for s in d if all((g, j) in d[s] for g in ("DMT", "PLA") for j in (0, k)))
        res = {}
        for m in METRICS:
            dmt = [d[s][("DMT", k)][m] - d[s][("DMT", 0)][m] for s in subs]
            pla = [d[s][("PLA", k)][m] - d[s][("PLA", 0)][m] for s in subs]
            res[m] = paired(dmt, pla)
        res["subjects"] = subs
        out[f"post{k}_change_DMT_vs_PLA"] = res
    return out


def main():
    settings = json.loads((OUTPUTS / "onset_settings.json").read_text())
    report = {}
    for name, det in settings.items():
        path = OUTPUTS / f"onset_counts_{name}.json"
        if not path.exists():
            continue
        rs = rows(name)
        rep = {"detector": {k: v for k, v in det.items() if k not in ("template_file", "datasets")},
               "imagery_positive_control": imagery(rs)}
        if name.startswith("zenodo"):
            rep["zenodo"] = zenodo(rs)
        else:
            rep["osf"] = osf(rs)
        report[name] = rep
    (OUTPUTS / "onset_partB_report.json").write_text(json.dumps(report, indent=1))
    for name, rep in report.items():
        print(name, json.dumps(rep["imagery_positive_control"]["events_per_min_real_vs_surrogate"]))
        if "zenodo" in rep:
            z = rep["zenodo"]
            for c, v in z["per_condition"].items():
                print(f"  {c} n={v['n']} " + " ".join(f"{m}={v[m]['mean']:.3f}" for m in METRICS))
            for key in ("DMT_vs_EC", "EO_vs_EC", "DMT_vs_EO"):
                for m in ("rate", "rate_minus_surr", "score_minus_surr"):
                    p = z[key][m]
                    print(f"  {key} {m}: diff {p['mean_diff']:+.3f} ({p['n_a_greater']}/{p['n']}) "
                          f"p_w={p['p_wilcoxon']:.3g} p_t={p['p_ttest']:.3g}")
        else:
            for k in (1, 2, 3):
                for m in ("rate", "rate_minus_surr", "score_minus_surr"):
                    p = rep["osf"][f"post{k}_change_DMT_vs_PLA"][m]
                    print(f"  post{k} {m}: DMT {p['mean_a']:+.3f} PLA {p['mean_b']:+.3f} "
                          f"({p['n_a_greater']}/{p['n']}) p_w={p['p_wilcoxon']:.3g}")


if __name__ == "__main__":
    main()
