"""Cut PsiConnect (OpenNeuro ds006110) EEG into 1 s windows on the 62 THINGS
channels it shares (all but FCz, its online reference), at 250 Hz.

ses-01 = baseline (sober), ses-02 = psilocybin 19 mg. Tasks are delimited by
paired start/end markers in events.tsv: Resting_state, Guided_meditation,
First_music, Video (cloud movie).

Output: data/processed/psiconnect_windows.npz
"""
import warnings

import mne
import numpy as np
import pandas as pd

from common import DATA, SFREQ, THINGS_CHANNELS, WINDOW, normalize_windows

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

CHANNELS = [c for c in THINGS_CHANNELS if c != "FCz"]
TASKS = {"Resting_state": "rest", "Guided_meditation": "meditation",
         "First_music": "music", "Video": "clouds"}
MAX_PTP_UV = 200.0
STEP = WINDOW // 2


def segments(events):
    out = {}
    for prefix, name in TASKS.items():
        on = events[events.trial_type.str.startswith(prefix)].onset.to_numpy()
        if len(on) >= 2:
            out[name] = (on[0], on[1])
    return out


def windows_for(vhdr, events):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    rename = {c: t for c in raw.ch_names for t in CHANNELS if c.lower() == t.lower()}
    raw.rename_channels(rename)
    raw.pick(CHANNELS)
    raw.filter(0.5, 45.0)
    raw.set_eeg_reference("average")
    raw.resample(SFREQ)
    data = raw.get_data() * 1e6
    res = []
    for task, (t0, t1) in segments(events).items():
        starts = np.arange(int(t0 * SFREQ), int(t1 * SFREQ) - WINDOW, STEP)
        win = np.stack([data[:, s:s + WINDOW] for s in starts])
        keep = (np.ptp(win, axis=-1).max(axis=1) < MAX_PTP_UV) & ~(win.std(axis=-1) < 0.5).any(axis=1)
        res.append((task, win[keep].astype(np.float32), starts[keep] / SFREQ - t0, len(starts)))
    return res


def main():
    root = DATA / "raw/psiconnect"
    xs, meta = [], []
    for vhdr in sorted(root.glob("sub-*/ses-*/eeg/*_eeg.vhdr")):
        sub, ses = vhdr.parts[-4], vhdr.parts[-3]
        ev_path = vhdr.with_name(vhdr.name.replace("_eeg.vhdr", "_events.tsv"))
        if not vhdr.with_suffix(".eeg").exists() or not ev_path.exists():
            continue
        try:
            res = windows_for(vhdr, pd.read_csv(ev_path, sep="\t"))
        except Exception as exc:
            print(sub, ses, "failed:", exc, flush=True)
            continue
        for task, w, t, total in res:
            print(f"{sub} {ses} {task}: kept {len(w)}/{total}", flush=True)
            if len(w):
                xs.append(normalize_windows(w))
                meta.append(pd.DataFrame({"subject": sub, "session": ses, "task": task, "t": t}))
    meta = pd.concat(meta, ignore_index=True)
    np.savez_compressed(DATA / "processed/psiconnect_windows.npz", x=np.concatenate(xs),
                        **{k: meta[k].to_numpy().astype(str if k != "t" else float) for k in ("subject", "session", "task", "t")})


if __name__ == "__main__":
    main()
