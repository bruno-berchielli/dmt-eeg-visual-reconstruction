"""Part A: do the THINGS-defined onset detectors find image onsets in an
independent lab (visual imagery dataset, Figshare 30227503, onsets known)?

Every detector variant (channel set x per-recording adaptation x detector)
is slid over each whole recording; per trial we keep the score time course
from -8 to +9 s around the marker (imagery onset), the score near the image
onset, and the score in matched null intervals (fixation and rest, windows
that contain no screen change). Also the imagery lab's own visual evoked
response as a template (leave-one-subject-out), as a fallback detector.

Pass 1 accumulates per-subject onset crops (VEP templates) and the raw
occipital ERP; pass 2 scores. Results: outputs/onset_imagery_scores.npz
(compact per-variant arrays) and outputs/onset_imagery_report.json via
onset_imagery_report.py.
"""
import itertools
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.signal import find_peaks

from common import MODELS, OUTPUTS, SFREQ, normalize_windows
from onset_lib import (CHANNEL_SETS, CONT, OnsetNet, STEP, adapt, circular_fir, clean_mask, load_cont,
                       mf_score, net_score, segments)

MODES = ["", "s", "t", "st"]
TAU = np.arange(-8.0, 9.0001, STEP / SFREQ)
REGION = (-6.4, -5.6)
IMAGE_ONSET = -6.0
NULLS = [(-8.3, -6.05), (4.85, 8.05)]
N_NULL = 3


def variants():
    for s, m in itertools.product(CHANNEL_SETS, MODES):
        for k in ("T200", "T500", "T1000", "net", "V125", "V250"):
            if k == "T500" and "t" in m:
                continue
            if k == "net" and "t" in m:
                continue
            yield s, m, k


def recordings():
    return sorted(CONT.glob("imagery_*.npz"))


def subject_of(path):
    return path.stem.split("_")[1]


def kind_len(k):
    return {"T200": 50, "T500": 125, "T1000": 250, "net": 125, "V125": 125, "V250": 250}[k]


def pass1(path):
    x_all, pieces, d = load_cont(path, CHANNEL_SETS["vi30"])
    onsets = d["onset"]
    out = {}
    good_all = clean_mask(x_all, pieces)
    for s, chans in CHANNEL_SETS.items():
        idx = [CHANNEL_SETS["vi30"].index(c) for c in chans]
        x = x_all[idx] - x_all[idx].mean(0, keepdims=True)
        for m in MODES:
            xa, _ = adapt(x, good_all, m)
            for L in (125, 250):
                i0 = np.round((onsets + IMAGE_ONSET) * SFREQ).astype(int)
                ok = [i for i in i0 if i >= 0 and i + L <= x.shape[1] and good_all[i:i + L].all()]
                w = normalize_windows(np.stack([xa[:, i:i + L] for i in ok])) if ok else np.zeros((1, len(chans), L))
                out[(s, m, L)] = (w.sum(0), len(ok))
    i0 = np.round((onsets + IMAGE_ONSET) * SFREQ).astype(int)
    ok = [i for i in i0 if i - 125 >= 0 and i + 375 <= x_all.shape[1] and good_all[i - 125:i + 375].all()]
    erp = np.stack([x_all[:, i - 125:i + 375] for i in ok]).mean(0) if ok else np.zeros((x_all.shape[0], 500))
    return subject_of(path), out, erp, len(ok)


def things_template(tpl, s, m, k, firs):
    t = tpl[f"{s}_{k}"]
    if "t" in m:
        t = circular_fir(t, firs)
    return t


def pass2(args):
    path = args
    vep, tpl = shared()
    torch.set_num_threads(2)
    sub = subject_of(path)
    x_all, pieces, d = load_cont(path, CHANNEL_SETS["vi30"])
    onsets = d["onset"]
    good = clean_mask(x_all, pieces)
    segs = segments(good)
    rng = np.random.default_rng(zlib.crc32(path.stem.encode()))
    nets = {}
    res = {}
    for s, chans in CHANNEL_SETS.items():
        idx = [CHANNEL_SETS["vi30"].index(c) for c in chans]
        x = x_all[idx] - x_all[idx].mean(0, keepdims=True)
        for m in MODES:
            xa, firs = adapt(x, good, m)
            for (vs, vm, k) in variants():
                if (vs, vm) != (s, m):
                    continue
                L = kind_len(k)
                if k == "net":
                    if s not in nets:
                        ck = torch.load(MODELS / f"onset_net_{s}.pt", map_location="cpu")
                        nets[s] = OnsetNet(len(chans), ck["n_times"])
                        nets[s].load_state_dict(ck["state_dict"])
                    starts = np.arange(0, xa.shape[1] - L + 1, STEP)
                    sc = np.full(xa.shape[1], np.nan, np.float32)
                    v = np.repeat(net_score(nets[s], xa, starts, L, "cpu"), STEP)
                    sc[:len(v)] = v[:len(sc)]
                else:
                    if k.startswith("V"):
                        tot, n = vep[(s, m, L)]["sum"], vep[(s, m, L)]["n"]
                        own, n_own = vep[(s, m, L)]["by_sub"][sub]
                        t = (tot - own) / (n - n_own)
                    else:
                        t = things_template(tpl, s, m, k, firs)
                    sc = np.full(xa.shape[1], np.nan, np.float32)
                    sc[:xa.shape[1] - L + 1] = mf_score(xa, t)
                win_ok = np.zeros(xa.shape[1], bool)
                for a, b in segs:
                    win_ok[a:b - max(L, SFREQ) + 1] = True
                sc[~win_ok] = np.nan
                res[(s, m, k)] = summarize(sc, onsets, L, k, rng)
    return sub, res


def summarize(sc, onsets, L, k, rng):
    n = len(sc)
    fine = 2 if k != "net" else STEP
    reg = np.arange(REGION[0], REGION[1] + 1e-9, fine / SFREQ)
    tc, rg, nl = [], [], []
    null_mask = np.zeros(n, bool)
    for o in onsets:
        i = np.round((o + TAU) * SFREQ).astype(int)
        tc.append(np.where((i >= 0) & (i < n), sc[np.clip(i, 0, n - 1)], np.nan))
        i = np.round((o + reg) * SFREQ).astype(int)
        rg.append(np.where((i >= 0) & (i < n), sc[np.clip(i, 0, n - 1)], np.nan))
        row = []
        for a, b in NULLS:
            ia, ib = int(round((o + a) * SFREQ)), int(round((o + b - L / SFREQ) * SFREQ))
            null_mask[max(ia, 0):max(ib + 1, 0)] = True
        cand = [(a, b - L / SFREQ - 0.2) for a, b in NULLS]
        for _ in range(N_NULL):
            a, b = cand[rng.integers(len(cand))]
            st = rng.uniform(a, b)
            i = np.round((o + st + np.arange(0, 0.2 + 1e-9, fine / SFREQ)) * SFREQ).astype(int)
            v = sc[np.clip(i, 0, n - 1)]
            row.append(np.nan if ((i < 0) | (i >= n)).any() or np.isnan(v).all() else np.nanmax(v))
        nl.append(row)
    null_mask &= ~np.isnan(sc)
    grid = np.zeros(n, bool)
    grid[::STEP] = True
    ss = np.where(grid & ~np.isnan(sc))[0]
    v = sc[ss]
    brk = np.where(np.diff(ss) > STEP)[0] + 1
    peaks, heights = [], []
    for seg_i, seg_v in zip(np.split(ss, brk), np.split(v, brk)):
        if len(seg_v) < 3:
            continue
        p, _ = find_peaks(seg_v, distance=int(0.5 * SFREQ / STEP))
        peaks.append(seg_i[p])
        heights.append(seg_v[p])
    peaks = np.concatenate(peaks) if peaks else np.array([], int)
    heights = np.concatenate(heights) if heights else np.array([])
    rel = np.full(len(peaks), np.nan)
    if len(peaks):
        on = np.round(onsets * SFREQ).astype(int)
        j = np.clip(np.searchsorted(on, peaks + 9 * SFREQ) - 1, 0, len(on) - 1)
        rel = (peaks - on[j]) / SFREQ
    tc = np.array(tc, np.float32)
    return {"tc_sum": np.nansum(tc, 0), "tc_n": (~np.isnan(tc)).sum(0), "region": np.array(rg, np.float16), "null": np.array(nl, np.float32),
            "peaks_rel": rel.astype(np.float32), "peaks_h": heights.astype(np.float32),
            "peaks_in_null": null_mask[peaks] if len(peaks) else np.array([], bool),
            "null_minutes": float((null_mask & grid).sum() * STEP / SFREQ / 60),
            "valid_minutes": float(grid[~np.isnan(sc)].sum() * STEP / SFREQ / 60)}


_SHARED = {}


def shared():
    if not _SHARED:
        z = np.load(OUTPUTS / "onset_imagery_pass1.npz")
        vep = {}
        for key in z.files:
            parts = key.split("|")
            if parts[0] == "erp" or parts[-1] == "n":
                continue
            s, m, L, sub = parts
            e = vep.setdefault((s, m, int(L)), {"sum": 0, "n": 0, "by_sub": {}})
            n = int(z[key + "|n"])
            e["by_sub"][sub] = (z[key], n)
            e["sum"] = e["sum"] + z[key]
            e["n"] += n
        _SHARED["vep"] = vep
        _SHARED["tpl"] = dict(np.load(MODELS / "onset_templates.npz"))
    return _SHARED["vep"], _SHARED["tpl"]


def main():
    recs = recordings()
    p1 = OUTPUTS / "onset_imagery_pass1.npz"
    if not p1.exists():
        with ProcessPoolExecutor(4) as ex:
            r1 = list(ex.map(pass1, recs))
        flat = {}
        for sub, out, erp, n in r1:
            for key, (sm, cnt) in out.items():
                flat.setdefault(key, {})
                a, c = flat[key].get(sub, (0, 0))
                flat[key][sub] = (a + sm, c + cnt)
        save = {}
        for (s, m, L), by in flat.items():
            for sub, (a, c) in by.items():
                save[f"{s}|{m}|{L}|{sub}"] = a
                save[f"{s}|{m}|{L}|{sub}|n"] = np.array(c)
        erps = {}
        for sub, out, erp, n in r1:
            a, c = erps.get(sub, (0, 0))
            erps[sub] = (a + erp * n, c + n)
        for sub, (a, c) in erps.items():
            save[f"erp|{sub}"] = a / max(c, 1)
            save[f"erp|{sub}|n"] = np.array(c)
        np.savez(p1, **save)
    with ProcessPoolExecutor(4) as ex:
        r2 = list(ex.map(pass2, recs))
    save = {}
    for i, (sub, res) in enumerate(r2):
        save[f"rec{i}|sub"] = np.array(sub)
        for (s, m, k), d in res.items():
            for f, v in d.items():
                save[f"rec{i}|{s}|{m}|{k}|{f}"] = np.asarray(v)
    np.savez_compressed(OUTPUTS / "onset_imagery_scores.npz", **save)
    print("done", len(r2), flush=True)


if __name__ == "__main__":
    main()
