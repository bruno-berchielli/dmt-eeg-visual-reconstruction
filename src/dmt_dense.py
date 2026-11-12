"""Dense, template-aligned decoding of the DMT recordings for the videos.

Instead of ~10 thresholded events per minute (dmt_events.py), decode a window
at EVERY local maximum of the onset detector (>= 0.5 s apart, no threshold):
about 8 times more phase-aligned windows per recording. The same is done on
a phase-randomized surrogate, whose mean is subtracted (what detector +
decoder return for the spectrum alone).

Reports the split-half agreement of the resulting trajectory (alternating
10 s blocks, so neighbouring windows never sit in different halves) for two
kernel widths, next to the thresholded-events version.

Stored per recording: t, v (real minus the surrogate's mean) and ts, vs (the
surrogate's own windows, centered), so the spectrum's time course can be
removed locally.

Output: data/processed/dmt_dense.pt, outputs/dmt_dense_report.json
"""
import json
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

import dmt_events_video as vid
from common import DATA, MODELS, OUTPUTS, SFREQ, SHARED_CHANNELS
from drug_direction import decode, starts_for
from event_decoders import EventDecoder
from onset_lib import CHANNEL_SETS, CONT, clean_mask, load_cont, segments, surrogate

BLOCK = 10.0


def block_halves(t, x, sigma, n_perm=1000, seed=0):
    clock = torch.arange(float(t.min()), float(t.max()) + vid.STEP, vid.STEP)
    half = (t // BLOCK).long() % 2

    def traj(sel, xs):
        w = torch.exp(-0.5 * ((t[sel][None] - clock[:, None]) / sigma) ** 2)
        return (w / w.sum(1, keepdim=True)) @ xs

    def same(u, v):
        return float(F.cosine_similarity(u - u.mean(0), v - v.mean(0), dim=1).mean())

    a, b = half == 0, half == 1
    ta = traj(a, x[a])
    obs = same(ta, traj(b, x[b]))
    # null: shift half B's vectors in time by whole blocks (keeps their autocorrelation)
    g = np.random.default_rng(seed)
    xb, nb = x[b], int(b.sum())
    null = [same(ta, traj(b, torch.roll(xb, int(g.integers(nb // 4, 3 * nb // 4)), 0))) for _ in range(n_perm)]
    return obs, float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1))


def process(path, tpl, idx, dec, min_windows=100):
    """Dense decoding of one recording, or None when it has too few windows."""
    x, pieces, _ = load_cont(path, SHARED_CHANNELS)
    st = starts_for(x, "peaks", tpl, idx)
    if len(st) < min_windows:
        return None
    rng = np.random.default_rng(zlib.crc32(path.stem.encode()))
    xs = surrogate(x, segments(clean_mask(x, pieces)), rng)
    ss = starts_for(xs, "peaks", tpl, idx)
    vs = decode(xs, ss, dec)
    v = decode(x, st, dec) - vs.mean(0)
    # surrogate windows keep the slow time course of the spectrum (phases are randomized per clean segment)
    return {"t": torch.from_numpy(st / SFREQ).float(), "v": v.half(),
            "ts": torch.from_numpy(ss / SFREQ).float(), "vs": (vs - vs.mean(0)).half()}


def main():
    tpl = np.load(MODELS / "onset_templates_multi.npz")["z14_T500"]
    idx = [SHARED_CHANNELS.index(c) for c in CHANNEL_SETS["z14"]]
    dec = EventDecoder.load("clip_multi")
    out, rows = {}, []
    for path in sorted(CONT.glob("dmt_*.npz")):
        _, subj, cond = path.stem.split("_")
        rec = process(path, tpl, idx, dec)
        if rec is None:
            continue
        out[path.stem] = rec
        t, v = rec["t"], rec["v"].float()
        row = {"s": subj, "c": cond, "n": len(t)}
        for sigma in (30.0, 15.0):
            row[f"r{int(sigma)}"], row[f"p{int(sigma)}"] = block_halves(t, v, sigma)
        rows.append(row)
        print(row, flush=True)
    torch.save(out, DATA / "processed/dmt_dense.pt")
    df = pd.DataFrame(rows)
    rep = {}
    for c, g in df.groupby("c"):
        rep[c] = {"n": len(g), "windows": float(g.n.median())}
        for k in ("30", "15"):
            rep[c][f"sigma{k}"] = {"mean": float(g[f"r{k}"].mean()), "p_group": float(wilcoxon(g[f"r{k}"], alternative="greater").pvalue),
                                    "n_p_below_0.05": int((g[f"p{k}"] < 0.05).sum())}
    (OUTPUTS / "dmt_dense_report.json").write_text(json.dumps({"group": rep, "rows": rows}, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
