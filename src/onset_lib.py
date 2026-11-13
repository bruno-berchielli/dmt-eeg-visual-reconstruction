"""Shared pieces for finding "seeing-like events" (image-onset-like responses)
in continuous EEG without knowing when they happen.

Two detectors, both defined from THINGS-EEG2 (or, as a fallback, from the
imagery lab's own visual evoked response):
  matched filter  normalized cross-correlation between a multichannel evoked
                  template and every window of the continuous signal, each
                  window normalized exactly like THINGS epochs
                  (common.normalize_windows);
  onset net       small CNN trained to say "an image onset happened at the
                  start of this window".

Continuous data are cached per recording (0.5-45 Hz, average reference,
250 Hz) in data/processed/onset_cont/. Optional per-recording adaptations,
all label-free: spatial whitening (THINGS is MVNN-whitened, targets are not)
and temporal AR prewhitening (as in gravitational-wave matched filtering,
the inner product is weighted by the recording's own noise spectrum).
"""
import re
import warnings

import mne
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.signal import find_peaks, lfilter, oaconvolve, resample_poly

from common import DATA, SFREQ, SHARED_CHANNELS, THINGS_CHANNELS, normalize_windows

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

CONT = DATA / "processed/onset_cont"
CONT.mkdir(parents=True, exist_ok=True)
VI_CHANNELS = [c for c in THINGS_CHANNELS if c in (
    "Fpz Fp1 Fp2 Fz F3 F4 F7 F8 FCz FC3 FC4 FT7 FT8 Cz C3 C4 T7 T8 CP3 CP4 "
    "TP7 TP8 Pz P3 P4 P7 P8 PO3 PO4 Oz O1 O2").split()]
OSF_CHANNELS = ["Oz", "Pz", "CPz", "Cz", "FCz"]
CHANNEL_SETS = {
    "vi30": VI_CHANNELS,
    "z14": [c for c in VI_CHANNELS if c in SHARED_CHANNELS],
    "z14post": ["P3", "Pz", "P4", "O1", "O2"],
    "o4": ["Pz", "Cz", "FCz", "Oz"],
}
MAX_PTP_UV = 200.0
STEP = 5


def _bdf_continuous(path, channels):
    raw = mne.io.read_raw_bdf(path, preload=True)
    raw.pick(channels)
    raw.filter(0.5, 45.0)
    raw.set_eeg_reference("average")
    raw.resample(SFREQ)
    return raw.get_data().astype(np.float32) * 1e6


def cache_imagery(path):
    sub, ses, task = re.search(r"(sub-\d+)_(ses-\d+)_task-(\w+)_eeg", path.name).groups()
    out = CONT / f"imagery_{sub}_{ses}_{task}.npz"
    if not out.exists():
        ev = pd.read_csv(str(path).replace("_eeg.bdf", "_events.tsv"), sep="\t")
        np.savez(out, x=_bdf_continuous(path, VI_CHANNELS).astype(np.float16), onset=ev.onset.to_numpy(),
                 channels=np.array(VI_CHANNELS))
    return out


def cache_dmt(path):
    m = re.match(r"[sS](\d+)[-_](DMT|EC|EO|Ec)", path.stem)
    out = CONT / f"dmt_S{int(m.group(1)):02d}_{m.group(2).upper()}.npz"
    if not out.exists():
        np.savez(out, x=_bdf_continuous(path, SHARED_CHANNELS), channels=np.array(SHARED_CHANNELS))
    return out


def cache_osf():
    for name in ("DMT", "PLA"):
        m = sio.loadmat(DATA / f"raw/osf_waves/data{name}.mat")[f"data{name}"]
        for s in range(m.shape[0]):
            for c in range(m.shape[1]):
                out = CONT / f"osf_{name}_s{s + 1:02d}_seg{c}.npz"
                if out.exists():
                    continue
                x = m[s, c]["trials"][:len(OSF_CHANNELS)].astype(np.float64)
                t = m[s, c]["time"].ravel()
                if np.isnan(x).any():
                    continue
                cuts = np.where(np.diff(t) > 0.0015)[0] + 1
                pieces = [resample_poly(p, 1, 4, axis=1).astype(np.float32)
                          for p in np.split(x, cuts, axis=1) if p.shape[1] >= 4000]
                np.savez(out, x=np.concatenate(pieces, axis=1), piece_len=np.array([p.shape[1] for p in pieces]),
                         channels=np.array(OSF_CHANNELS))


def load_cont(path, channels):
    d = np.load(path)
    names = list(d["channels"])
    x = d["x"].astype(np.float32)[[names.index(c) for c in channels]]
    x[~np.isfinite(x)] = 1e4
    x -= x.mean(0, keepdims=True)
    pieces = d["piece_len"] if "piece_len" in d else np.array([x.shape[1]])
    return x, pieces, d


def clean_mask(x, pieces, win=SFREQ, hop=SFREQ // 5, max_ptp=MAX_PTP_UV):
    """Boolean per sample: False inside any 1 s window (0.2 s hop) whose
    peak-to-peak reaches max_ptp or that has a flat channel, and at piece
    boundaries (OSF data are concatenated clean stretches)."""
    good = np.ones(x.shape[1], bool)
    for s in range(0, x.shape[1] - win + 1, hop):
        w = x[:, s:s + win]
        if np.ptp(w, axis=1).max() >= max_ptp or (w.std(axis=1) < 0.5).any():
            good[s:s + win] = False
    edges = np.cumsum(pieces)[:-1]
    good[np.r_[edges, edges - 1]] = False
    return good


def segments(good, min_len=3 * SFREQ):
    d = np.diff(np.r_[0, good.astype(np.int8), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_len]


def surrogate(x, segs, rng):
    """Phase randomization per clean segment with the same random phases on
    every channel: keeps each channel's power spectrum and the cross-channel
    covariance, destroys time-locked structure."""
    out = np.zeros_like(x)
    for a, b in segs:
        spec = np.fft.rfft(x[:, a:b].astype(np.float64), axis=1)
        ph = np.exp(2j * np.pi * rng.random(spec.shape[1]))
        ph[0] = 1
        if (b - a) % 2 == 0:
            ph[-1] = 1
        out[:, a:b] = np.fft.irfft(spec * ph[None], n=b - a, axis=1)
    return out


def spatial_whitener(x, good, shrink=0.1):
    xs = x[:, good].astype(np.float64)
    xs = xs / xs.std(axis=1, keepdims=True)
    cov = xs @ xs.T / xs.shape[1]
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / len(cov) * np.eye(len(cov))
    vals, vecs = np.linalg.eigh(cov)
    return (vecs @ np.diag(vals ** -0.5) @ vecs.T / x[:, good].std(axis=1)[None]).astype(np.float32)


def ar_whitener(x, good, order=10):
    """Per-channel AR(order) prediction-error FIR, Yule-Walker on clean data."""
    firs = []
    for ch in x:
        v = ch[good].astype(np.float64)
        v = v - v.mean()
        r = np.array([v[:len(v) - k] @ v[k:] for k in range(order + 1)]) / len(v)
        R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
        phi = np.linalg.solve(R + 1e-9 * r[0] * np.eye(order), r[1:order + 1])
        firs.append(np.r_[1.0, -phi])
    return np.array(firs, dtype=np.float32)


def apply_fir(x, firs):
    return np.stack([lfilter(f, [1.0], ch) for f, ch in zip(firs, x)]).astype(np.float32)


def circular_fir(template, firs):
    """Filter a periodic template (THINGS RSVP cycle) circularly."""
    L = template.shape[1]
    out = np.stack([np.real(np.fft.ifft(np.fft.fft(t) * np.fft.fft(f, L))) for t, f in zip(template, firs)])
    out = out - out.mean(axis=1, keepdims=True)
    return out.astype(np.float32)


def adapt(x, good, mode):
    """mode: subset of 's' (spatial whitening) and 't' (AR prewhitening)."""
    if good.sum() < 10 * SFREQ:
        good = np.ones_like(good)
    firs = None
    if "t" in mode:
        firs = ar_whitener(x, good)
        x = apply_fir(x, firs)
    if "s" in mode:
        x = spatial_whitener(x, good) @ x
        x -= x.mean(0, keepdims=True)
    return x, firs


def adapt_fit(x, good, mode):
    if good.sum() < 10 * SFREQ:
        good = np.ones_like(good)
    firs = ar_whitener(x, good) if "t" in mode else None
    xf = apply_fir(x, firs) if firs is not None else x
    W = spatial_whitener(xf, good) if "s" in mode else None
    return firs, W


def adapt_apply(x, firs, W):
    if firs is not None:
        x = apply_fir(x, firs)
    if W is not None:
        x = W @ x
        x = x - x.mean(0, keepdims=True)
    return x


def mf_score(x, template):
    """Normalized cross-correlation at every start sample; each window gets
    per-channel demean and z-score (as normalize_windows; the per-sample
    average reference is applied to x beforehand). Returns len N-L+1."""
    C, L = template.shape
    t = template - template.mean(axis=1, keepdims=True)
    num = np.zeros(x.shape[1] - L + 1)
    for c in range(C):
        xc = x[c].astype(np.float64)
        corr = oaconvolve(xc, t[c][::-1].astype(np.float64), mode="valid")
        cs = np.cumsum(np.r_[0, xc])
        cs2 = np.cumsum(np.r_[0, xc ** 2])
        mu = (cs[L:] - cs[:-L]) / L
        sd = np.sqrt(np.maximum((cs2[L:] - cs2[:-L]) / L - mu ** 2, 0)) + 1e-6
        num += corr / sd
    return num / (np.linalg.norm(t) * np.sqrt(C * L))


class OnsetNet(nn.Module):
    """EEGNet-sized: temporal conv, depthwise spatial conv, pooling, logit."""

    def __init__(self, n_channels, n_times=125, f1=8, d=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, f1, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 5)),
            nn.Dropout(0.25),
            nn.Conv2d(f1 * d, f1 * d, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(f1 * d * (n_times // 5), 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1)


@torch.no_grad()
def net_score(model, x, starts, L, dev, batch=4096):
    model.eval()
    view = np.lib.stride_tricks.sliding_window_view(x, L, axis=1)
    out = np.empty(len(starts), np.float32)
    for i in range(0, len(starts), batch):
        w = normalize_windows(np.ascontiguousarray(view[:, starts[i:i + batch]].transpose(1, 0, 2)))
        out[i:i + batch] = model(torch.from_numpy(w).to(dev)).float().cpu().numpy()
    return out


def valid_starts(segs, L, step=STEP):
    return np.concatenate([np.arange(a, b - max(L, SFREQ) + 1, step) for a, b in segs]) if segs else np.array([], int)


def count_events(score, starts, segs, thr, refractory_s=0.5, step=STEP):
    """Local maxima above thr, at least refractory_s apart, found separately
    inside each clean segment. Returns event start samples."""
    ev = []
    for a, b in segs:
        sel = (starts >= a) & (starts < b)
        if sel.sum() < 3:
            continue
        p, _ = find_peaks(score[sel], height=thr, distance=max(1, int(refractory_s * SFREQ / step)))
        ev.append(starts[sel][p])
    return np.concatenate(ev) if ev else np.array([], int)
