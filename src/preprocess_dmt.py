"""Cut the Zenodo 3992359 DMT / EC / EO recordings into 1 s windows on the
21 channels shared with THINGS-EEG2, at 250 Hz.

Output: data/processed/dmt_windows.npz with
  x       (n_windows, 21, 250) float32, already normalized like THINGS epochs
  subject, condition, t_start (s from recording start), ptp_uv
"""
import re
import warnings

import mne
import numpy as np

from common import DATA, SFREQ, SHARED_CHANNELS, WINDOW, normalize_windows

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

MAX_PTP_UV = 200.0
STEP = WINDOW // 2


def windows_for(path):
    raw = mne.io.read_raw_bdf(path, preload=True)
    raw.pick(SHARED_CHANNELS)
    raw.filter(0.5, 45.0)
    raw.set_eeg_reference("average")
    raw.resample(SFREQ)
    data = raw.get_data() * 1e6
    starts = np.arange(0, data.shape[1] - WINDOW, STEP)
    win = np.stack([data[:, s:s + WINDOW] for s in starts])
    ptp = np.ptp(win, axis=-1).max(axis=1)
    flat = (win.std(axis=-1) < 0.5).any(axis=1)
    keep = (ptp < MAX_PTP_UV) & ~flat
    return win[keep].astype(np.float32), starts[keep] / SFREQ, ptp[keep]


def main():
    xs, subj, cond, t0, ptps = [], [], [], [], []
    summary = []
    for path in sorted((DATA / "raw/dmt").glob("*.bdf")):
        m = re.match(r"[sS](\d+)[-_](DMT|EC|EO|Ec)", path.stem)
        s, c = f"S{int(m.group(1)):02d}", m.group(2).upper()
        w, t, ptp = windows_for(path)
        total = int(mne.io.read_raw_bdf(path).times[-1] // (STEP / SFREQ))
        summary.append((s, c, len(w), total))
        print(f"{s} {c}: kept {len(w)}/{total} windows", flush=True)
        if len(w) == 0:
            continue
        xs.append(normalize_windows(w))
        subj += [s] * len(w)
        cond += [c] * len(w)
        t0.append(t)
        ptps.append(ptp)
    np.savez_compressed(
        DATA / "processed/dmt_windows.npz",
        x=np.concatenate(xs), subject=np.array(subj), condition=np.array(cond),
        t_start=np.concatenate(t0), ptp_uv=np.concatenate(ptps))


if __name__ == "__main__":
    main()
