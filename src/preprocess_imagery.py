"""Visual imagery dataset (Figshare 10.6084/m9.figshare.30227503, CC BY 4.0).

Per trial (17.1 s): fixation 3 s, image shown 4 s, mask, black screen with
imagery 4 s, rest 4 s. The event marker is the imagery onset, verified on
the occipital evoked responses: image onset at -6.0 s, image offset at
-2.0 s, imagery onset at 0, rest cue at +4.25 s.

Windows (1 s, 250 Hz, 30 channels shared with THINGS):
  perception  -6.0 .. -5.0 s  (first second of the image, like THINGS epochs)
  imagery     0.5 .. 4.0 s    (six windows, 0.5 s step; skips the black-screen onset)
  rest        4.75 .. 7.75 s  (three windows)

Output: data/processed/imagery_windows.npz
"""
import re
import warnings

import mne
import numpy as np
import pandas as pd

from common import DATA, SFREQ, THINGS_CHANNELS, WINDOW, normalize_windows

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

VI_CHANNELS = ("Fpz Fp1 Fp2 Fz F3 F4 F7 F8 FCz FC3 FC4 FT7 FT8 Cz C3 C4 T7 T8 CP3 CP4 "
               "TP7 TP8 Pz P3 P4 P7 P8 PO3 PO4 Oz O1 O2").split()
CHANNELS = [c for c in THINGS_CHANNELS if c in VI_CHANNELS]
PHASES = {"perception": [-6.0], "imagery": list(np.arange(0.5, 3.01, 0.5)), "rest": [4.75, 5.75, 6.75]}
MAX_PTP_UV = 200.0


def main():
    xs, meta = [], []
    for bdf in sorted((DATA / "raw/imagery/extracted").glob("sub-*/ses-*/eeg/*_eeg.bdf")):
        sub, ses, task = re.search(r"(sub-\d+)_(ses-\d+)_task-(\w+)_eeg", bdf.name).groups()
        ev = pd.read_csv(str(bdf).replace("_eeg.bdf", "_events.tsv"), sep="\t")
        raw = mne.io.read_raw_bdf(bdf, preload=True)
        raw.pick(CHANNELS)
        raw.filter(0.5, 45.0)
        raw.set_eeg_reference("average")
        raw.resample(SFREQ)
        data = raw.get_data() * 1e6
        kept = 0
        for _, row in ev.iterrows():
            for phase, starts in PHASES.items():
                for st in starts:
                    i = int(round((row.onset + st) * SFREQ))
                    if i < 0 or i + WINDOW > data.shape[1]:
                        continue
                    w = data[:, i:i + WINDOW]
                    if np.ptp(w, axis=-1).max() >= MAX_PTP_UV:
                        continue
                    xs.append(w.astype(np.float32))
                    meta.append((sub, ses, task, row["type"], int(row["trial"]), phase, float(st)))
                    kept += 1
        print(f"{sub} {ses} {task}: {kept} windows", flush=True)
    m = pd.DataFrame(meta, columns=["subject", "session", "task", "image", "trial", "phase", "start"])
    np.savez_compressed(DATA / "processed/imagery_windows.npz", x=normalize_windows(np.stack(xs)),
                        **{k: m[k].to_numpy().astype(str) for k in ("subject", "session", "task", "image", "phase")},
                        trial=m.trial.to_numpy(), start=m.start.to_numpy())


if __name__ == "__main__":
    main()
