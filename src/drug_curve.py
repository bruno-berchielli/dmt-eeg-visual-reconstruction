"""Does the EEG state follow the kinetics expected for inhaled DMT (fast
rise, slow decay) along each recording?

No inhalation markers exist in the files, so time is counted from the start
of the recording. Per 10 s bin, from the whole recording (no artifact
rejection; medians over 2 s windows to resist bursts):
  alpha   8-12 Hz share of 1-30 Hz power, posterior channels
  delta   1-4 Hz share
  hf      30-45 Hz share (muscle proxy)
  amp     median peak-to-peak (movement proxy)
Each is expressed relative to the same person's eyes-closed recording.

Model comparison per recording, for log alpha: variance explained by
  drug     difference of exponentials, onset / rise / decay on a grid
  mirror   the same family reversed in time (slow rise, fast decay)
  linear   straight drift
The same fits on the eyes-closed recordings are the no-drug control.

Output: outputs/drug_curve.json, data/processed/drug_curve.npz
"""
import json

import numpy as np
from scipy.signal import welch
from scipy.stats import wilcoxon

from common import DATA, OUTPUTS, SFREQ, SHARED_CHANNELS
from onset_lib import CONT, load_cont

POSTERIOR = [c for c in SHARED_CHANNELS if c[0] in "OP"]
BIN = 10


def state(path):
    x, _, _ = load_cont(path, SHARED_CHANNELS)
    post = [SHARED_CHANNELS.index(c) for c in POSTERIOR]
    w = 2 * SFREQ
    rows = []
    for s in range(0, x.shape[1] - w + 1, w):
        seg = x[:, s:s + w]
        f, p = welch(seg, SFREQ, nperseg=w)
        band = lambda a, b, ch=slice(None): p[ch][:, (f >= a) & (f < b)].sum(1).mean()
        tot = band(1, 30)
        rows.append([band(8, 12, post) / band(1, 30, post), band(1, 4) / tot, band(30, 45) / tot,
                     np.median(np.ptp(seg, axis=1))])
    r = np.array(rows)
    n = len(r) // (BIN // 2)
    return np.median(r[:n * (BIN // 2)].reshape(n, BIN // 2, 4), axis=1)


def curve(t, t0, rise, decay):
    u = np.clip(t - t0, 0, None)
    k = np.exp(-u / decay) - np.exp(-u / rise)
    return k / max(k.max(), 1e-9)


def r2(y, k):
    X = np.c_[np.ones_like(k), k]
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - res.var() / y.var()


def fits(y):
    t = np.arange(len(y)) * BIN + BIN / 2
    best = {"drug": (-1, None), "mirror": (-1, None)}
    for t0 in range(-120, 181, 20):
        for rise in (10, 20, 40, 80):
            for decay in (90, 180, 360, 720):
                k = curve(t, t0, rise, decay)
                if k.std() < 1e-6:
                    continue
                for name, kk in (("drug", k), ("mirror", k[::-1])):
                    v = r2(y, kk)
                    if v > best[name][0]:
                        best[name] = (v, (t0, rise, decay))
    out = {name: float(v) for name, (v, _) in best.items()}
    out["linear"] = float(r2(y, t))
    out["drug_params"] = best["drug"][1]
    k = curve(t, *best["drug"][1])
    out["drug_sign"] = float(np.sign(np.linalg.lstsq(np.c_[np.ones_like(k), k], y, rcond=None)[0][1]))
    out["peak_s"] = float(t[k.argmax()])
    return out


def main():
    rep, store = {"per_recording": {}}, {}
    for path in sorted(CONT.glob("dmt_*_DMT.npz")):
        subj = path.stem.split("_")[1]
        ec_path = CONT / f"dmt_{subj}_EC.npz"
        d = state(path)
        store[f"{subj}_DMT"] = d
        rec = {"minutes": len(d) * BIN / 60, "DMT": fits(np.log(d[:, 0]))}
        if ec_path.exists():
            e = state(ec_path)
            store[f"{subj}_EC"] = e
            rec["EC"] = fits(np.log(e[:, 0]))
            rec["alpha_DMT_over_EC"] = float(np.median(d[:, 0]) / np.median(e[:, 0]))
        rep["per_recording"][subj] = rec
    for c in ("DMT", "EC"):
        rs = [r[c] for r in rep["per_recording"].values() if c in r]
        g = {k: float(np.median([r[k] for r in rs])) for k in ("drug", "mirror", "linear")}
        g["n"] = len(rs)
        g["p_drug_gt_mirror"] = float(wilcoxon([r["drug"] - r["mirror"] for r in rs], alternative="greater").pvalue)
        g["drug_sign_negative"] = int(sum(r["drug_sign"] < 0 for r in rs))
        g["median_peak_s"] = float(np.median([r["peak_s"] for r in rs]))
        rep[c] = g
    (OUTPUTS / "drug_curve.json").write_text(json.dumps(rep, indent=1))
    np.savez(DATA / "processed/drug_curve.npz", **store)
    print(json.dumps({k: rep[k] for k in ("DMT", "EC")}, indent=1))

    def group(c, col, n_bins):
        m = [np.log(v[:n_bins, col] / np.median(store[k[:3] + "_EC"][:, col]))
             for k, v in store.items() if k.endswith(c) and k[:3] + "_EC" in store and len(v) >= n_bins]
        return np.median(np.stack(m), 0), len(m)

    for name, col in (("alpha", 0), ("delta", 1), ("hf", 2), ("amp", 3)):
        for c, n_bins in (("DMT", 36), ("EC", 24)):
            g, n = group(c, col, n_bins)
            print(f"{name} {c} (n={n}, log ratio to own EC median, 30 s steps):",
                  np.round(g.reshape(-1, 3).mean(1), 2).tolist())

if __name__ == "__main__":
    main()
