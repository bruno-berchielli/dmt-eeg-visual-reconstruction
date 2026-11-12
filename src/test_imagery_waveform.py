"""Within-lab check on the imagery dataset with raw waveforms instead of band
power: per subject (ses-01, all three tasks), cross-validated logistic
regression on baseline-corrected 0.5-20 Hz waveforms at 50 Hz.

Compares imagery (0-4 s and 1-4 s after imagery onset), perception (first
second of the image) and imagery restricted to frontal channels (Fp1, Fp2,
F7, F8), whose decodability would point to eye movements rather than visual
cortex.
"""
import json
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, OUTPUTS
from preprocess_imagery import CHANNELS

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
SF = 50
FRONTAL = ("Fp1", "Fp2", "F7", "F8")


def valid(d, onsets):
    return np.array([int((o - 6.2) * SF) >= 0 and int((o + 4) * SF) <= d.shape[1] for o in onsets])


def features(d, onsets, a, b, ch):
    n = int(round((b - a) * SF))
    out = []
    for o in onsets:
        i0 = int(round((o + a) * SF))
        seg = d[ch, i0:i0 + n]
        base = d[ch, int((o - 0.2) * SF):int(o * SF)].mean(1, keepdims=True)
        out.append((seg - base).reshape(-1))
    return np.array(out)


def main():
    windows = {"imagery_0-4s": (0, 4, None), "imagery_1-4s": (1, 4, None),
               "perception": (-6, -5, None), "imagery_0-4s_frontal_only": (0, 4, FRONTAL)}
    res = {k: [] for k in windows}
    for f in sorted((DATA / "raw/imagery/extracted").glob("sub-*/ses-01/eeg/*_eeg.bdf")):
        raw = mne.io.read_raw_bdf(f, preload=True)
        raw.pick(CHANNELS)
        raw.set_eeg_reference("average")
        raw.filter(0.5, 20)
        raw.resample(SF)
        ev = pd.read_csv(str(f).replace("_eeg.bdf", "_events.tsv"), sep="\t")
        d = raw.get_data() * 1e6
        ev = ev[valid(d, ev.onset)]
        y = ev["type"].to_numpy()
        chance = 1 / len(set(y))
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        for k, (a, b, ch) in windows.items():
            idx = list(range(len(CHANNELS))) if ch is None else [CHANNELS.index(c) for c in ch]
            res[k].append(cross_val_score(clf, features(d, ev.onset, a, b, idx), y, cv=cv).mean() - chance)
    report = {k: {"acc_minus_chance_mean": float(np.mean(v)), "n": len(v),
                  "p_above_chance": float(wilcoxon(v, alternative="greater").pvalue)} for k, v in res.items()}
    (OUTPUTS / "imagery_waveform_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
