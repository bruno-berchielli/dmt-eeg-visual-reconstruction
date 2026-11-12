"""Band-power features for the phenomenology predictor, computed identically
for PsiConnect (psilocybin) and the Zenodo DMT dataset: the 21 shared
channels, 0.5-45 Hz, average reference over those 21, 1 s windows at 250 Hz,
artifact windows dropped, log power per channel and band (median over
windows).

Output: data/processed/pheno_features.npz
  psi_*  one row per (subject, session, task)
  dmt_*  one row per (subject, condition), plus per-window DMT rows for
         time-resolved prediction
"""
import re
import warnings

import mne
import numpy as np
import pandas as pd

from common import DATA, SFREQ, SHARED_CHANNELS, WINDOW
from preprocess_psiconnect import segments

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
FREQS = np.fft.rfftfreq(WINDOW, 1 / SFREQ)
MAX_PTP_UV = 200.0


def prepare(raw):
    raw.rename_channels({c: t for c in raw.ch_names for t in SHARED_CHANNELS if c.lower() == t.lower()})
    raw.pick(SHARED_CHANNELS)
    raw.filter(0.5, 45.0)
    raw.set_eeg_reference("average")
    raw.resample(SFREQ)
    return raw.get_data() * 1e6


def window_power(data, t0=None, t1=None):
    """Log band power per good window: (n_windows, 21, 5) and window start times."""
    i0 = 0 if t0 is None else int(t0 * SFREQ)
    i1 = data.shape[1] if t1 is None else int(t1 * SFREQ)
    starts = np.arange(i0, i1 - WINDOW, WINDOW // 2)
    if len(starts) == 0:
        return np.zeros((0, len(SHARED_CHANNELS), len(BANDS))), np.zeros(0)
    win = np.stack([data[:, s:s + WINDOW] for s in starts])
    keep = (np.ptp(win, axis=-1).max(axis=1) < MAX_PTP_UV) & ~(win.std(axis=-1) < 0.5).any(axis=1)
    win = win[keep] - win[keep].mean(-1, keepdims=True)
    p = np.abs(np.fft.rfft(win * np.hanning(WINDOW), axis=-1)) ** 2
    bp = np.stack([p[..., (FREQS >= lo) & (FREQS < hi)].mean(-1) for lo, hi in BANDS], -1)
    return np.log(bp + 1e-12), (starts[keep] - i0) / SFREQ


def main():
    psi_rows, psi_feats = [], []
    for vhdr in sorted((DATA / "raw/psiconnect").glob("sub-*/ses-*/eeg/*_eeg.vhdr")):
        sub, ses = vhdr.parts[-4], vhdr.parts[-3]
        ev = pd.read_csv(vhdr.with_name(vhdr.name.replace("_eeg.vhdr", "_events.tsv")), sep="\t")
        try:
            data = prepare(mne.io.read_raw_brainvision(vhdr, preload=True))
        except Exception as exc:
            print(sub, ses, "failed:", exc, flush=True)
            continue
        for task, (t0, t1) in segments(ev).items():
            bp, _ = window_power(data, t0, t1)
            if len(bp) >= 30:
                psi_rows.append((sub, ses, task, len(bp)))
                psi_feats.append(np.median(bp, 0))
        print(sub, ses, flush=True)

    dmt_rows, dmt_feats, win_rows, win_feats = [], [], [], []
    for path in sorted((DATA / "raw/dmt").glob("*.bdf")):
        m = re.match(r"[sS](\d+)[-_](DMT|EC|EO|Ec)", path.stem)
        s, c = f"S{int(m.group(1)):02d}", m.group(2).upper()
        bp, t = window_power(prepare(mne.io.read_raw_bdf(path, preload=True)))
        if len(bp) >= 30:
            dmt_rows.append((s, c, len(bp)))
            dmt_feats.append(np.median(bp, 0))
            if c == "DMT":
                win_rows += [(s, float(x)) for x in t]
                win_feats.append(bp)
        print(s, c, len(bp), flush=True)

    np.savez_compressed(
        DATA / "processed/pheno_features.npz",
        psi_feat=np.stack(psi_feats), psi_subject=np.array([r[0] for r in psi_rows]),
        psi_session=np.array([r[1] for r in psi_rows]), psi_task=np.array([r[2] for r in psi_rows]),
        dmt_feat=np.stack(dmt_feats), dmt_subject=np.array([r[0] for r in dmt_rows]),
        dmt_condition=np.array([r[1] for r in dmt_rows]),
        win_feat=np.concatenate(win_feats), win_subject=np.array([r[0] for r in win_rows]),
        win_t=np.array([r[1] for r in win_rows]))


if __name__ == "__main__":
    main()
