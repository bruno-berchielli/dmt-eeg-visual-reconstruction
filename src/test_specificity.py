"""Specificity panel: do different spectral perturbations of the same
eyes-closed EEG decode to different things, and is DMT distinct from them?

For each subject, EC windows are modified in the frequency domain:
  alpha_down   8-13 Hz amplitude x0.5
  fast_up      20-45 Hz amplitude x2
  theta_up     4-7 Hz amplitude x2
  flatten      1/f slope flattened (gain grows with frequency)
  muscle       added bursts of 25-45 Hz noise on temporal/frontal channels
  dmt_like     full DMT spectral transplant (per subject)
Real contrasts: DMT-EC and EO-EC. Reports pairwise cosine between group
directions and the top THINGS concepts per perturbation, and renders one
image per perturbation and decoder.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, OUTPUTS, SFREQ, SHARED_CHANNELS, WINDOW, normalize_windows
from decode import ConceptSpace
from embed_dmt import MODEL_NAMES, load_decoders
from test_spectral_transplant import transplant

FREQS = np.fft.rfftfreq(WINDOW, 1 / SFREQ)


def band_gain(lo, hi, g):
    gain = np.ones_like(FREQS)
    gain[(FREQS >= lo) & (FREQS <= hi)] = g
    return gain


def apply_gain(x, gain):
    spec = np.fft.rfft(x, axis=-1) * gain
    return normalize_windows(np.fft.irfft(spec, n=x.shape[-1], axis=-1).astype(np.float32))


def muscle(x, rng):
    idx = [SHARED_CHANNELS.index(c) for c in ("T7", "T8", "F7", "F8", "Fp1", "Fp2")]
    noise = rng.standard_normal((len(x), len(idx), WINDOW))
    noise = np.fft.irfft(np.fft.rfft(noise, axis=-1) * band_gain(25, 45, 1)[None, None] *
                         ((FREQS >= 25) & (FREQS <= 45)), n=WINDOW, axis=-1)
    y = x.copy()
    y[:, idx] += 1.5 * noise / noise.std()
    return normalize_windows(y.astype(np.float32))


PERTURB = {
    "alpha_down": lambda x, rng, dmt: apply_gain(x, band_gain(8, 13, 0.5)),
    "fast_up": lambda x, rng, dmt: apply_gain(x, band_gain(20, 45, 2.0)),
    "theta_up": lambda x, rng, dmt: apply_gain(x, band_gain(4, 7, 2.0)),
    "flatten": lambda x, rng, dmt: apply_gain(x, np.clip(FREQS / 10, 0.3, 4.5) ** 0.5),
    "muscle": lambda x, rng, dmt: muscle(x, rng),
    "dmt_like": lambda x, rng, dmt: transplant(x, dmt),
}


def main():
    w = np.load(DATA / "processed/dmt_windows.npz")
    x, subj, cond = w["x"], w["subject"], w["condition"]
    decoders = load_decoders("08")
    cs = ConceptSpace()
    rng = np.random.default_rng(0)

    def n(s, c):
        return int(((subj == s) & (cond == c)).sum())

    subjects = [s for s in sorted(set(subj)) if min(n(s, "DMT"), n(s, "EC"), n(s, "EO")) >= 30]
    report, groups = {}, {}
    for m in MODEL_NAMES:
        dec = decoders[m]
        per = {k: [] for k in ["DMT", "EO", *PERTURB]}
        for s in subjects:
            ec, dmt = x[(subj == s) & (cond == "EC")], x[(subj == s) & (cond == "DMT")]
            base = dec(ec).mean(0)
            per["DMT"].append(dec(dmt).mean(0) - base)
            per["EO"].append(dec(x[(subj == s) & (cond == "EO")]).mean(0) - base)
            for k, f in PERTURB.items():
                per[k].append(dec(f(ec, rng, dmt)).mean(0) - base)
        g = {k: torch.stack(v).mean(0) for k, v in per.items()}
        groups[m] = g
        names = list(g)
        cos = {a: {b: round(F.cosine_similarity(g[a], g[b], dim=0).item(), 2) for b in names} for a in names}
        report[m] = {"cosine": cos, "top": {k: cs.top(v, 8) for k, v in g.items()},
                     "norm": {k: round(v.norm().item(), 3) for k, v in g.items()}}
    torch.save(groups, DATA / "processed/specificity_groups.pt")
    (OUTPUTS / "specificity_report.json").write_text(json.dumps(report, indent=1))
    for m in MODEL_NAMES:
        print("==", m)
        names = list(report[m]["cosine"])
        print("cos".ljust(11) + "".join(k[:9].rjust(10) for k in names))
        for a in names:
            print(a[:10].ljust(11) + "".join(f"{report[m]['cosine'][a][b]:10.2f}" for b in names))
        for k, v in report[m]["top"].items():
            print(f"  {k}: {', '.join(v)}")


if __name__ == "__main__":
    main()
