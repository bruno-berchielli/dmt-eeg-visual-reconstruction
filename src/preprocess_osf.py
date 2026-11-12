"""Cut the OSF wujgp data (Alamia et al., eLife 2020; IV DMT vs placebo, eyes
closed, 1-45 Hz, 1000 Hz) into 1 s windows at 250 Hz.

The .mat files carry no channel labels. The paper analyses "five central
mid-line electrodes (Oz to FCz)" as the first five rows; we assume those are
Oz, Pz, CPz, Cz, FCz (posterior-to-anterior alpha gradient is consistent with
this order). Condition 0 is the pre-injection baseline, 1-3 are successive
post-injection segments.

Output: data/processed/osf_windows.npz
"""
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from common import DATA, WINDOW, normalize_windows

OSF_CHANNELS = ["Oz", "Pz", "CPz", "Cz", "FCz"]
SEGMENTS = ["pre", "post1", "post2", "post3"]
MAX_PTP_UV = 200.0
STEP = WINDOW // 2


def main():
    xs, drug, subj, seg, t = [], [], [], [], []
    for name in ("DMT", "PLA"):
        m = sio.loadmat(DATA / f"raw/osf_waves/data{name}.mat")[f"data{name}"]
        for s in range(m.shape[0]):
            for c in range(m.shape[1]):
                x = m[s, c]["trials"][:len(OSF_CHANNELS)].astype(np.float64)
                if np.isnan(x).any():
                    continue
                time = m[s, c]["time"].ravel()[::4]
                x = resample_poly(x, 1, 4, axis=1)
                starts = np.arange(0, x.shape[1] - WINDOW, STEP)
                win = np.stack([x[:, i:i + WINDOW] for i in starts])
                keep = np.ptp(win, axis=-1).max(axis=1) < MAX_PTP_UV
                print(f"{name} s{s + 1:02d} {SEGMENTS[c]}: kept {keep.sum()}/{len(keep)}", flush=True)
                xs.append(normalize_windows(win[keep].astype(np.float32)))
                n = int(keep.sum())
                drug += [name] * n
                subj += [f"s{s + 1:02d}"] * n
                seg += [SEGMENTS[c]] * n
                t.append(time[starts[keep]])
    np.savez_compressed(DATA / "processed/osf_windows.npz", x=np.concatenate(xs), drug=np.array(drug),
                        subject=np.array(subj), segment=np.array(seg), t=np.concatenate(t))


if __name__ == "__main__":
    main()
