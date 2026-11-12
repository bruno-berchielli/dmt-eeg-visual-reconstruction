"""Part A report: score time courses, image-onset latency, per-subject AUC
(image onset vs matched null intervals), hit rate at a threshold fixed for
3 false alarms per minute in fixation/rest, for every detector variant.

Then freezes the detector settings for Part B (outputs/onset_settings.json):
for each target montage (Zenodo -> z14/z14post, OSF -> o4) the variant with
the highest mean AUC, separately among THINGS-defined detectors and among
the imagery-lab VEP templates. Nothing here looks at drug data.
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.metrics import roc_auc_score

from common import MODELS, OUTPUTS, SFREQ
from onset_imagery import NULLS, REGION, STEP, TAU, kind_len

FA_PER_MIN = 3.0
TRANSITIONS = {"image_on": -6.0, "image_off_mask": -2.0, "black_screen_imagery": 0.0, "rest_cue": 4.25,
               "next_fixation": 8.1}


def load():
    z = np.load(OUTPUTS / "onset_imagery_scores.npz")
    data = defaultdict(lambda: defaultdict(list))
    subs = {}
    for key in z.files:
        parts = key.split("|")
        if parts[1] == "sub":
            subs[parts[0]] = str(z[key])
    for key in z.files:
        parts = key.split("|")
        if parts[1] == "sub":
            continue
        rec, s, m, k, f = parts
        data[(s, m, k)][f].append((subs[rec], z[key]))
    return data


def analyse(d, k):
    L = kind_len(k) / SFREQ
    fine = (2 if k != "net" else STEP) / SFREQ
    reg_t = np.arange(REGION[0], REGION[1] + 1e-9, fine)
    subs = sorted({s for s, _ in d["tc_sum"]})
    tc = {s: [0, 0] for s in subs}
    for (s, a), (_, b) in zip(d["tc_sum"], d["tc_n"]):
        tc[s][0] = tc[s][0] + a
        tc[s][1] = tc[s][1] + b
    tcs = np.array([tc[s][0] / np.maximum(tc[s][1], 1) for s in subs])
    null_pts = np.zeros(len(TAU), bool)
    for a, b in NULLS:
        null_pts |= (TAU >= a) & (TAU <= b - L)
    sub_null = np.nanmean(tcs[:, null_pts], 1)
    group = np.nanmean(tcs, 0)
    null_sd = np.nanstd(group[null_pts])
    out = {"n_subjects": len(subs), "null_mean": float(np.nanmean(sub_null)), "null_sd_group_tc": float(null_sd),
           "transitions": {}}
    for name, T in TRANSITIONS.items():
        w = (TAU >= T - 0.4) & (TAU <= T + 0.4)
        j = np.where(w)[0][np.nanargmax(group[w])]
        diff = tcs[:, j] - sub_null
        out["transitions"][name] = {
            "peak_latency_s": round(float(TAU[j]), 3), "peak": float(group[j]),
            "z_vs_null": float((group[j] - np.nanmean(group[null_pts])) / null_sd),
            "t_across_subjects": float(ttest_1samp(diff, 0).statistic), "p_wilcoxon": float(wilcoxon(diff).pvalue)}
    regs = defaultdict(list)
    for s, r in d["region"]:
        regs[s].append(r.astype(np.float32))
    regs = {s: np.concatenate(v) for s, v in regs.items()}
    sub_reg = np.array([np.nanmean(regs[s], 0) for s in subs])
    lat = float(reg_t[np.nanargmax(np.nanmean(sub_reg, 0))])
    out["image_onset_latency_s"] = round(lat, 3)
    nulls = defaultdict(list)
    for s, v in d["null"]:
        nulls[s].append(v.ravel())
    aucs, lats = [], []
    for i, s in enumerate(subs):
        l = float(reg_t[np.nanargmax(np.nanmean(np.delete(sub_reg, i, 0), 0))])
        lats.append(l)
        sel = np.abs(reg_t - l) <= 0.1 + 1e-9
        pos = regs[s][:, sel]
        pos = pos[~np.isnan(pos).all(1)]
        pos = np.nanmax(pos, 1)
        neg = np.concatenate(nulls[s])
        neg = neg[~np.isnan(neg)]
        if len(pos) < 5 or len(neg) < 5:
            continue
        aucs.append(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    aucs = np.array(aucs)
    out["auc_per_subject"] = np.round(aucs, 3).tolist()
    out["auc_mean"] = float(aucs.mean())
    out["auc_sd"] = float(aucs.std())
    out["auc_n_above_0.5"] = int((aucs > 0.5).sum())
    out["auc_p_wilcoxon_vs_0.5"] = float(wilcoxon(aucs - 0.5).pvalue)
    out["loso_latency_range_s"] = [min(lats), max(lats)]
    h = np.concatenate([v for _, v in d["peaks_h"]])
    rel = np.concatenate([v for _, v in d["peaks_rel"]])
    innull = np.concatenate([v for _, v in d["peaks_in_null"]]).astype(bool)
    null_min = sum(float(v) for _, v in d["null_minutes"])
    valid_min = sum(float(v) for _, v in d["valid_minutes"])
    nh = np.sort(h[innull])[::-1]
    kth = int(round(FA_PER_MIN * null_min))
    thr = float(nh[kth]) if kth < len(nh) else float(nh[-1])
    n_trials = sum(int((~np.isnan(r.astype(np.float32)).all(1)).sum()) for _, r in d["region"])
    out["threshold_at_3_fa_per_min"] = thr
    out["null_minutes"] = null_min
    out["hit_rate_image_onset_at_thr"] = float(((np.abs(rel - lat) <= 0.1) & (h >= thr)).sum() / n_trials)
    out["events_per_min_whole_recording_at_thr"] = float((h >= thr).sum() / valid_min)
    out["events_per_min_near_each_transition_at_thr"] = {
        n: float(((np.abs(rel - T) <= 0.3) & (h >= thr)).sum() / n_trials) for n, T in TRANSITIONS.items()}
    out["group_time_course_20ms"] = np.round(group, 4).tolist()
    return out


def erp_latencies():
    z = np.load(OUTPUTS / "onset_imagery_pass1.npz")
    from onset_lib import VI_CHANNELS
    occ = [VI_CHANNELS.index(c) for c in ("O1", "Oz", "O2", "PO3", "PO4")]
    erps = np.array([z[k] for k in z.files if k.startswith("erp|") and not k.endswith("|n")
                     and int(z[k + "|n"]) > 0])
    t = (np.arange(500) - 125) / SFREQ
    occ_erp = erps[:, occ].mean((0, 1))
    occ_erp = occ_erp - occ_erp[t < 0].mean()
    gfp = erps.mean(0).std(0)
    base = gfp[t < 0]
    above = np.where((t > 0) & (gfp > base.mean() + 3 * base.std()))[0]
    w1, w2 = (t >= 0.05) & (t <= 0.15), (t >= 0.12) & (t <= 0.25)
    return {"occipital_P1_latency_s": float(t[w1][np.argmax(occ_erp[w1])]),
            "occipital_N1_latency_s": float(t[w2][np.argmin(occ_erp[w2])]),
            "gfp_first_exceeds_baseline_3sd_s": float(t[above[0]]) if len(above) else None,
            "note": "t = 0 is the nominal image onset, marker - 6.0 s"}


def main():
    data = load()
    report = {"erp_relative_to_nominal_image_onset": erp_latencies(), "variants": {}}
    for (s, m, k), d in sorted(data.items()):
        name = f"{s}|{m or 'raw'}|{k}"
        report["variants"][name] = analyse(d, k)
        r = report["variants"][name]
        print(f"{name:22s} AUC {r['auc_mean']:.3f} ({r['auc_n_above_0.5']}/{len(r['auc_per_subject'])}) "
              f"lat {r['image_onset_latency_s']:+.3f} hit {r['hit_rate_image_onset_at_thr']:.2f} "
              f"thr {r['threshold_at_3_fa_per_min']:.3f} img z {r['transitions']['image_on']['z_vs_null']:.1f}",
              flush=True)
    settings = {}
    pass1 = np.load(OUTPUTS / "onset_imagery_pass1.npz")
    vep = {}
    for target, sets, pats in (("zenodo", ("z14", "z14post"), ["imagery_*.npz", "dmt_*.npz"]),
                               ("osf", ("o4",), ["imagery_*.npz", "osf_*.npz"])):
        for family, kinds in (("things", ("T200", "T500", "T1000", "net")), ("vep", ("V125", "V250"))):
            cands = {n: r for n, r in report["variants"].items()
                     if n.split("|")[0] in sets and n.split("|")[2] in kinds}
            best = max(cands, key=lambda n: cands[n]["auc_mean"])
            s, m, k = best.split("|")
            m = "" if m == "raw" else m
            det = {"set": s, "mode": m, "kind": k, "thr": cands[best]["threshold_at_3_fa_per_min"],
                   "datasets": pats, "imagery_auc_mean": cands[best]["auc_mean"],
                   "imagery_hit_rate": cands[best]["hit_rate_image_onset_at_thr"]}
            if k.startswith("T"):
                det.update(template_file=str(MODELS / "onset_templates.npz"), template_key=f"{s}_{k}")
            elif k.startswith("V"):
                L = kind_len(k)
                keys = [f for f in pass1.files if f.startswith(f"{s}|{m}|{L}|") and not f.endswith("|n")]
                tot = sum(pass1[f] for f in keys)
                n = sum(int(pass1[f + "|n"]) for f in keys)
                vep[f"{s}_{m}_{k}"] = (tot / n).astype(np.float32)
                det.update(template_file=str(MODELS / "onset_vep_templates.npz"), template_key=f"{s}_{m}_{k}")
            settings[f"{target}_{family}"] = det
    np.savez(MODELS / "onset_vep_templates.npz", **vep)
    (OUTPUTS / "onset_settings.json").write_text(json.dumps(settings, indent=1))
    (OUTPUTS / "onset_imagery_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(settings, indent=1))
    print(json.dumps(report["erp_relative_to_nominal_image_onset"], indent=1))


if __name__ == "__main__":
    main()
