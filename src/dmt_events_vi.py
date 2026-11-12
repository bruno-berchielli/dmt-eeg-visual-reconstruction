"""Event-locked decoding of the Zenodo DMT recordings with the model trained
on the imagery dataset's perception phase (vi_model.py): single image onsets,
22 people, 14 shared channels. Same logic as dmt_events.py: threshold from a
phase-randomized surrogate of each recording, events above it, decode the
window starting at each event, and do the same on a second surrogate and at
random clean times.

Decoder output: scores for the 10 imagery images (3 geometric figures,
3 animals, 4 objects).

Output: data/processed/dmt_events_vi.pt
"""
import zlib

import numpy as np
import torch

from common import DATA, SFREQ, SHARED_CHANNELS, WINDOW, normalize_windows
from onset_lib import CHANNEL_SETS, CONT, clean_mask, count_events, load_cont, mf_score, segments, surrogate, valid_starts
from vi_model import VIModel

QUANTILE = 0.995
MAX_RANDOM = 2000


def main():
    m = VIModel()
    idx = [SHARED_CHANNELS.index(c) for c in CHANNEL_SETS["z14"]]
    out = {}

    def score(x21):
        x = x21[idx]
        x = x - x.mean(0, keepdims=True)
        sc = np.full(x.shape[1], -np.inf, np.float32)
        sc[:x.shape[1] - m.template.shape[1] + 1] = mf_score(x, m.template)
        return sc, x

    def decode(x14, starts):
        if len(starts) == 0:
            return None
        return m.scores(normalize_windows(np.stack([x14[:, s:s + WINDOW] for s in starts])))

    for path in sorted(CONT.glob("dmt_*.npz")):
        _, subj, cond = path.stem.split("_")
        x, pieces, _ = load_cont(path, SHARED_CHANNELS)
        segs = segments(clean_mask(x, pieces))
        grid = valid_starts(segs, m.template.shape[1])
        if len(grid) < 60 * SFREQ / 5:
            continue
        minutes = len(grid) * 5 / SFREQ / 60
        rng = np.random.default_rng(zlib.crc32(path.stem.encode()))
        sc, x14 = score(x)
        sc1, _ = score(surrogate(x, segs, rng))
        sc2, s14 = score(surrogate(x, segs, rng))
        thr = float(np.quantile(sc1[grid], QUANTILE))
        ev = count_events(sc[grid], grid, segs, thr)
        ev_s = count_events(sc2[grid], grid, segs, thr)
        rnd = rng.choice(grid, min(MAX_RANDOM, len(grid)), replace=False)
        d_ev, d_s, d_r = decode(x14, ev), decode(s14, ev_s), decode(x14, rnd)
        out[path.stem] = {
            "subject": subj, "condition": cond, "minutes": minutes, "rate": len(ev) / minutes,
            "rate_surrogate": len(ev_s) / minutes, "events_s": ev / SFREQ,
            "events": None if d_ev is None else torch.from_numpy(d_ev).float(),
            "surrogate_mean": None if d_s is None else torch.from_numpy(d_s.mean(0)).float(),
            "random_mean": torch.from_numpy(d_r.mean(0)).float(),
        }
        print(f"{path.stem}: {minutes:.1f} min, {len(ev) / minutes:.1f} ev/min vs surrogate {len(ev_s) / minutes:.1f}",
              flush=True)
    torch.save(out, DATA / "processed/dmt_events_vi.pt")


if __name__ == "__main__":
    main()
