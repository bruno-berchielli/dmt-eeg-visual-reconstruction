"""Part B: count seeing-like events per minute with the detector settings
frozen in Part A (outputs/onset_settings.json, written by
onset_imagery_report.py before any drug data were scored).

For every recording: clean stretches only (no 1 s window with peak-to-peak
>= 200 uV or a flat channel), events = local maxima of the detector score
above the frozen threshold, >= 0.5 s apart. Each recording is compared with
phase-randomized surrogates of its own clean stretches (same power spectrum
per channel, same random phases on all channels so the channel covariance
is kept), because a spectral change alone moves the false-alarm rate.
Per-recording adaptations (whitening) are fitted on the real data and
applied unchanged to its surrogates.

Datasets: imagery lab (positive control: real screen changes every few
seconds), Zenodo DMT / EC / EO, OSF IV DMT vs placebo (pre and 3 post
segments). Outputs: outputs/onset_counts_<detector>.json (per recording)
and data/processed/onset_events_<detector>.npz (event times, s from the
start of the cached continuous recording).
"""
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from common import DATA, MODELS, OUTPUTS, SFREQ
from onset_lib import (CHANNEL_SETS, CONT, STEP, OnsetNet, adapt_apply, adapt_fit, circular_fir, clean_mask,
                       count_events, load_cont, mf_score, net_score, segments, surrogate)

N_SURR = 3
KIND_LEN = {"T200": 50, "T500": 125, "T1000": 250, "net": 125, "V125": 125, "V250": 250}


def detector_scores(x, firs, W, det, tpl):
    xa = adapt_apply(x, firs, W)
    L = KIND_LEN[det["kind"]]
    if det["kind"] == "net":
        torch.set_num_threads(2)
        ck = torch.load(MODELS / f"onset_net_{det['set']}.pt", map_location="cpu")
        model = OnsetNet(len(ck["channels"]), ck["n_times"])
        model.load_state_dict(ck["state_dict"])
        starts = np.arange(0, xa.shape[1] - L + 1, STEP)
        sc = np.full(xa.shape[1], np.nan, np.float32)
        v = np.repeat(net_score(model, xa, starts, L, "cpu"), STEP)
        sc[:len(v)] = v[:len(sc)]
        return sc
    t = tpl
    if det["kind"].startswith("T") and firs is not None:
        t = circular_fir(t, firs)
    sc = np.full(xa.shape[1], np.nan, np.float32)
    sc[:xa.shape[1] - L + 1] = mf_score(xa, t)
    return sc


def run(args):
    path, det = args
    chans = CHANNEL_SETS[det["set"]]
    x_full, pieces, _ = load_cont(path, list(np.load(path)["channels"]))
    good = clean_mask(x_full, pieces)
    segs = segments(good)
    names = list(np.load(path)["channels"])
    x = x_full[[names.index(c) for c in chans]]
    x = x - x.mean(0, keepdims=True)
    tpl = np.load(det["template_file"])[det["template_key"]] if det["kind"] != "net" else None
    L = KIND_LEN[det["kind"]]
    out = {"recording": path.stem, "clean_minutes": 0.0, "total_minutes": x.shape[1] / SFREQ / 60}
    if not segs:
        return out, np.array([])
    firs, W = adapt_fit(x, good, det["mode"])
    grid = np.concatenate([np.arange(a, b - max(L, SFREQ) + 1, STEP) for a, b in segs])
    minutes = len(grid) * STEP / SFREQ / 60

    def measure(xx):
        sc = detector_scores(xx, firs, W, det, tpl)
        ev = count_events(sc[grid], grid, segs, det["thr"])
        return ev, float(np.nanmean(sc[grid]))

    ev, mscore = measure(x)
    rng = np.random.default_rng(zlib.crc32(path.stem.encode()))
    s_rates, s_scores = [], []
    for _ in range(N_SURR):
        sev, ssc = measure(surrogate(x, segs, rng))
        s_rates.append(len(sev) / minutes)
        s_scores.append(ssc)
    out.update(clean_minutes=minutes, events=int(len(ev)), rate=len(ev) / minutes, mean_score=mscore,
               surr_rate=float(np.mean(s_rates)), surr_mean_score=float(np.mean(s_scores)),
               rate_minus_surr=len(ev) / minutes - float(np.mean(s_rates)))
    return out, ev / SFREQ


def main():
    settings = json.loads((OUTPUTS / "onset_settings.json").read_text())
    names = sys.argv[1:] or list(settings)
    for name in names:
        det = settings[name]
        jobs = []
        for pattern in det["datasets"]:
            jobs += [(p, det) for p in sorted(CONT.glob(pattern))]
        with ProcessPoolExecutor(4) as ex:
            res = list(ex.map(run, jobs))
        (OUTPUTS / f"onset_counts_{name}.json").write_text(json.dumps([r for r, _ in res], indent=0))
        np.savez(DATA / f"processed/onset_events_{name}.npz", **{r["recording"]: e for r, e in res})
        print(name, "done", len(res), flush=True)


if __name__ == "__main__":
    main()
