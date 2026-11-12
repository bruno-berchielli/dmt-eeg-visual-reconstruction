"""Do the decoded vectors move in a consistent direction as the drug effect
rises and falls within each DMT recording?

Drug level d(t): minus log of posterior alpha relative to the same person's
eyes-closed recording, per 10 s bin (drug_curve.py), measured rather than
assumed from a kinetic curve (the files have no inhalation marker and start
with the effect already installed). Muscle proxy m(t): log 30-45 Hz share.

Windows decoded with the five-subject ridge, two ways:
  all     every clean window on a 0.5 s grid. The decoder is linear in the
          waveform, so without phase alignment its mean barely depends on
          the state; kept as the expected-null reference.
  peaks   windows starting at local maxima of the onset detector (>= 0.5 s
          apart, no threshold), i.e. phase-aligned to the template.

Per recording: v = a + b_d * d + b_m * m (regressors z-scored). Tests:
  halves        cosine between b_d from odd and even 20 s blocks
  across        do people share the b_d direction (sign-flip null)?
  vs_muscle     cosine between each person's b_d and b_m
  vs_contrast   cosine between b_d and that person's DMT - eyes-closed mean
  scales        size of the drug effect on the vectors vs questionnaires

Output: outputs/drug_direction.json, data/processed/drug_direction.pt
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from scipy.stats import spearmanr, wilcoxon

from common import DATA, MODELS, OUTPUTS, SFREQ, SHARED_CHANNELS, WINDOW, normalize_windows
from decode import ConceptSpace, sign_flip_null
from dmt_events import detector_score
from drug_curve import BIN
from event_decoders import EventDecoder
from onset_lib import CHANNEL_SETS, CONT, clean_mask, load_cont, segments, valid_starts


def starts_for(x, mode, tpl, idx):
    segs = segments(clean_mask(x, np.array([x.shape[1]])))
    grid = valid_starts(segs, tpl.shape[1])
    grid = grid[grid + WINDOW <= x.shape[1]]
    if mode == "all":
        return grid[::25]
    score = detector_score(x, tpl, idx)[grid]
    out = []
    for a, b in segs:
        sel = np.flatnonzero((grid >= a) & (grid < b))
        if len(sel) >= 3:
            p, _ = find_peaks(score[sel], distance=SFREQ // 2 // 5)
            out.append(grid[sel][p])
    return np.concatenate(out) if out else np.array([], int)


def decode(x, starts, dec):
    return torch.from_numpy(dec(normalize_windows(np.stack([x[:, s:s + WINDOW] for s in starts])))).float()


def level(curve, ec, col, starts):
    t = (np.arange(len(curve)) + 0.5) * BIN
    return np.interp(starts / SFREQ, t, np.log(curve[:, col] / np.median(ec[:, col])))


def slopes(v, regs):
    X = torch.from_numpy(np.c_[np.ones(len(v)), regs]).float()
    return torch.linalg.lstsq(X, v).solution[1:]


def z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def main():
    tpl = np.load(MODELS / "onset_templates_multi.npz")["z14_T500"]
    idx = [SHARED_CHANNELS.index(c) for c in CHANNEL_SETS["z14"]]
    dec = EventDecoder.load("clip_multi")
    curves = np.load(DATA / "processed/drug_curve.npz")
    sc = pd.read_csv(DATA / "raw/dmt/scales_results.csv", index_col=0, decimal=",")
    sc.index = [f"S{int(''.join(filter(str.isdigit, str(s)))):02d}" for s in sc.index]
    sc = sc.apply(pd.to_numeric, errors="coerce")
    cs = ConceptSpace()
    rep, store = {}, {}
    for mode in ("all", "peaks"):
        rows = []
        for path in sorted(CONT.glob("dmt_*_DMT.npz")):
            subj = path.stem.split("_")[1]
            if f"{subj}_EC" not in curves:
                continue
            x, _, _ = load_cont(path, SHARED_CHANNELS)
            st = starts_for(x, mode, tpl, idx)
            if len(st) < 100:
                continue
            v = decode(x, st, dec)
            d = z(-level(curves[f"{subj}_DMT"], curves[f"{subj}_EC"], 0, st))
            m = z(level(curves[f"{subj}_DMT"], curves[f"{subj}_EC"], 2, st))
            b = slopes(v, np.c_[d, m])
            block = (st // (20 * SFREQ)) % 2
            ha, hb = (slopes(v[block == k], np.c_[d, m][block == k])[0] for k in (0, 1))
            xe, _, _ = load_cont(CONT / f"dmt_{subj}_EC.npz", SHARED_CHANNELS)
            se = starts_for(xe, mode, tpl, idx)
            contrast = v.mean(0) - decode(xe, se, dec).mean(0) if len(se) >= 50 else None
            rows.append({"s": subj, "n": len(st), "b_d": b[0], "b_m": b[1], "b_d_alone": slopes(v, d[:, None])[0],
                         "halves": float(F.cosine_similarity(ha, hb, dim=0)),
                         "vs_muscle": float(F.cosine_similarity(b[0], b[1], dim=0)),
                         "vs_contrast": None if contrast is None else float(F.cosine_similarity(b[0], contrast, dim=0)),
                         "contrast": contrast,
                         "size": float(b[0].norm() / v.std(0).norm()), "corr_d_m": float(np.corrcoef(d, m)[0, 1])})
            print(mode, subj, len(st), round(rows[-1]["halves"], 2), flush=True)
        df = pd.DataFrame(rows).set_index("s")
        B = torch.stack(list(df.b_d))
        obs, null, p = sign_flip_null(B)
        obs_m, _, p_m = sign_flip_null(torch.stack(list(df.b_m)))
        obs_a, _, p_a = sign_flip_null(torch.stack(list(df.b_d_alone)))
        cons = df.dropna(subset=["vs_contrast"])
        obs_c, _, p_c = sign_flip_null(torch.stack(list(cons.contrast)))
        r = {"n": len(df), "windows_per_recording": float(df.n.median()),
             "halves": {"median": float(df.halves.median()), "positive": int((df.halves > 0).sum()),
                        "p": float(wilcoxon(df.halves, alternative="greater").pvalue)},
             "across_drug": {"consistency": obs, "null": null, "p": p},
             "across_drug_without_muscle_regressor": {"consistency": obs_a, "p": p_a},
             "across_muscle": {"consistency": obs_m, "p": p_m},
             "across_DMT_minus_EC": {"n": len(cons), "consistency": obs_c, "p": p_c},
             "vs_muscle": {"median": float(df.vs_muscle.median()), "p": float(wilcoxon(df.vs_muscle).pvalue)},
             "vs_contrast": {"n": len(cons), "median": float(cons.vs_contrast.median()),
                             "p": float(wilcoxon(cons.vs_contrast.astype(float), alternative="greater").pvalue)},
             "corr_drug_muscle_median": float(df.corr_d_m.median()),
             "group_drug_vs_group_muscle": float(F.cosine_similarity(B.mean(0), torch.stack(list(df.b_m)).mean(0), dim=0)),
             "top_concepts_drug": cs.top(B.mean(0), 12), "top_concepts_muscle": cs.top(torch.stack(list(df.b_m)).mean(0), 12),
             "top_concepts_DMT_minus_EC": cs.top(torch.stack(list(cons.contrast)).mean(0), 12), "scales": {}}
        for name in ("5D-Total", "5D-ElementaryImagery", "5D-ComplexImagery"):
            q = spearmanr(df["size"], sc[name].reindex(df.index))
            r["scales"][name] = {"rho": float(q.correlation), "p": float(q.pvalue)}
        rep[mode] = r
        store[mode] = {"subjects": list(df.index), "b_d": B, "b_m": torch.stack(list(df.b_m)),
                       "contrast_subjects": list(cons.index), "contrast": torch.stack(list(cons.contrast))}
    (OUTPUTS / "drug_direction.json").write_text(json.dumps(rep, indent=1))
    torch.save(store, DATA / "processed/drug_direction.pt")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
