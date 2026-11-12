"""Event-locked decoding of the Zenodo DMT recordings.

For each recording (DMT, eyes closed, eyes open):
  1. slide the THINGS matched-filter template (first 200 ms of the average
     response to an image onset, 14 channels) over the clean stretches;
  2. make a phase-randomized surrogate of the same stretches (same spectrum
     and channel covariance, no time-locked structure) and set the threshold
     at a high quantile of ITS score, so the false-alarm rate is fixed per
     recording whatever its spectrum;
  3. events = local maxima above that threshold, >= 0.5 s apart; a second
     surrogate gives the event rate expected from the spectrum alone;
  4. decode the 1 s window starting at each event with the jitter-tolerant
     decoders (content and low-level), and do the same at the surrogate's
     own events (what detector + decoder return for structureless data) and
     at random clean times.

Usage: python dmt_events.py T200|T500 [multi]
"multi" uses the template and content decoder fitted on five THINGS subjects
(things_multi.py) instead of sub-08 alone.

Output: data/processed/dmt_events_<kind>[multi].pt (per recording: rates,
event times, mean embeddings for events / surrogate events / random times).
"""
import sys
import zlib

import numpy as np
import torch

from common import DATA, MODELS, SFREQ, SHARED_CHANNELS, WINDOW, normalize_windows
from event_decoders import EventDecoder
from onset_lib import CHANNEL_SETS, CONT, clean_mask, count_events, load_cont, mf_score, segments, surrogate, valid_starts

QUANTILE = 0.995
MAX_RANDOM = 2000


def detector_score(x21, tpl, idx):
    x = x21[idx]
    x = x - x.mean(0, keepdims=True)
    sc = np.full(x21.shape[1], -np.inf, np.float32)
    sc[:x.shape[1] - tpl.shape[1] + 1] = mf_score(x, tpl)
    return sc


def decode_at(x21, starts, decoders):
    if len(starts) == 0:
        return {k: None for k in decoders}
    win = normalize_windows(np.stack([x21[:, s:s + WINDOW] for s in starts]))
    return {k: d(win) for k, d in decoders.items()}


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "T200"
    multi = len(sys.argv) > 2 and sys.argv[2] == "multi"
    tpl = np.load(MODELS / ("onset_templates_multi.npz" if multi else "onset_templates.npz"))[f"z14_{kind}"]
    idx = [SHARED_CHANNELS.index(c) for c in CHANNEL_SETS["z14"]]
    decoders = {"clip": EventDecoder.load("clip_multi" if multi else "clip"), "lowlevel": EventDecoder.load("lowlevel")}
    out = {}
    for path in sorted(CONT.glob("dmt_*.npz")):
        _, subj, cond = path.stem.split("_")
        x, pieces, _ = load_cont(path, SHARED_CHANNELS)
        segs = segments(clean_mask(x, pieces))
        grid = valid_starts(segs, tpl.shape[1])
        if len(grid) < 60 * SFREQ / 5:
            print(path.stem, "skipped (clean data < 1 min)", flush=True)
            continue
        minutes = len(grid) * 5 / SFREQ / 60
        rng = np.random.default_rng(zlib.crc32(path.stem.encode()))
        s1, s2 = surrogate(x, segs, rng), surrogate(x, segs, rng)
        thr = float(np.quantile(detector_score(s1, tpl, idx)[grid], QUANTILE))
        ev = count_events(detector_score(x, tpl, idx)[grid], grid, segs, thr)
        ev_s = count_events(detector_score(s2, tpl, idx)[grid], grid, segs, thr)
        rnd = rng.choice(grid, min(MAX_RANDOM, len(grid)), replace=False)
        rec = {"subject": subj, "condition": cond, "minutes": minutes, "threshold": thr,
               "rate": len(ev) / minutes, "rate_surrogate": len(ev_s) / minutes, "events_s": ev / SFREQ}
        for name, starts, data in (("events", ev, x), ("surrogate", ev_s, s2), ("random", rnd, x)):
            emb = decode_at(data, starts, decoders)
            for k, v in emb.items():
                rec[f"{name}_{k}_mean"] = None if v is None else torch.from_numpy(v.mean(0)).float()
                if name == "events" and v is not None:
                    rec[f"events_{k}"] = torch.from_numpy(v).half()
        out[path.stem] = rec
        print(f"{path.stem}: {minutes:.1f} min, {rec['rate']:.1f} ev/min vs surrogate {rec['rate_surrogate']:.1f}",
              flush=True)
    torch.save(out, DATA / f"processed/dmt_events_{kind}{'multi' if multi else ''}.pt")


if __name__ == "__main__":
    main()
