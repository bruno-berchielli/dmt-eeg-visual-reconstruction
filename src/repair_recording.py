"""Recover a recording that the artifact rule throws away almost entirely
because of one or two bad electrodes.

For each condition of the subject: 1 Hz high-pass (slow drifts), find the
channels that alone drive most rejected windows, rebuild them from their
neighbours (spherical splines, standard 10-20 positions), re-reference to
the average. The result is cached as subject "<id>r" (e.g. S27r), leaving
the original untouched, and added to the state curves (drug_curve.npz) and
to the dense decoding (dmt_dense.pt) so the video scripts can use it.

Usage: python repair_recording.py S27
"""
import sys

import mne
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

import dmt_dense
from common import DATA, MODELS, SFREQ, SHARED_CHANNELS
from drug_curve import state
from event_decoders import EventDecoder
from onset_lib import CHANNEL_SETS, CONT, MAX_PTP_UV, clean_mask, load_cont

SHARE = 0.25


def bad_channels(x):
    counts, n = np.zeros(len(x)), 0
    for a in range(0, x.shape[1] - SFREQ + 1, SFREQ):
        ptp = np.ptp(x[:, a:a + SFREQ], axis=1)
        if ptp.max() >= MAX_PTP_UV:
            n += 1
            counts += ptp >= MAX_PTP_UV
    return [SHARED_CHANNELS[i] for i in np.flatnonzero(counts >= SHARE * max(n, 1))] if n else []


def repair(path, out):
    x, pieces, _ = load_cont(path, SHARED_CHANNELS)
    before = clean_mask(x, pieces).mean()
    x = sosfiltfilt(butter(4, 1.0, "hp", fs=SFREQ, output="sos"), x.astype(np.float64), axis=1)
    bads = bad_channels(x)
    if bads and len(bads) <= 4:
        mne.set_log_level("ERROR")
        raw = mne.io.RawArray(x * 1e-6, mne.create_info(list(SHARED_CHANNELS), SFREQ, "eeg"))
        raw.set_montage("standard_1020")
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True)
        x = raw.get_data() * 1e6
    x = (x - x.mean(0, keepdims=True)).astype(np.float32)
    np.savez(out, x=x, channels=np.array(SHARED_CHANNELS))
    after = clean_mask(x, np.array([x.shape[1]])).mean()
    print(f"{path.stem}: clean {before:.0%} -> {after:.0%}, rebuilt {bads}", flush=True)


def main():
    subj = sys.argv[1]
    tpl = np.load(MODELS / "onset_templates_multi.npz")["z14_T500"]
    idx = [SHARED_CHANNELS.index(c) for c in CHANNEL_SETS["z14"]]
    dec = EventDecoder.load("clip_multi")
    curves = dict(np.load(DATA / "processed/drug_curve.npz"))
    dense = torch.load(DATA / "processed/dmt_dense.pt", weights_only=False)
    for cond in ("DMT", "EC", "EO"):
        src = CONT / f"dmt_{subj}_{cond}.npz"
        if not src.exists():
            continue
        out = CONT / f"dmt_{subj}r_{cond}.npz"
        repair(src, out)
        curves[f"{subj}r_{cond}"] = state(out)
        rec = dmt_dense.process(out, tpl, idx, dec)
        print(f"  dense windows: {0 if rec is None else len(rec['t'])}", flush=True)
        if rec is not None:
            dense[out.stem] = rec
    np.savez(DATA / "processed/drug_curve.npz", **curves)
    torch.save(dense, DATA / "processed/dmt_dense.pt")


if __name__ == "__main__":
    main()
