"""Per-recording channel whitening (Euclidean alignment).

THINGS-EEG2 was preprocessed with multivariate noise normalization, i.e. its
channels are already whitened per session; target datasets are not. Whitening
each target recording with its own channel covariance removes that mismatch
without labels.
"""
import numpy as np

from common import normalize_windows


def channel_cov(x):
    return np.einsum("nct,ndt->cd", x, x, optimize=True) / (x.shape[0] * x.shape[-1])


def whiten(x, shrink=0.1):
    """x: (n, channels, time) windows from ONE recording."""
    cov = channel_cov(x.astype(np.float64))
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / len(cov) * np.eye(len(cov))
    vals, vecs = np.linalg.eigh(cov)
    w = vecs @ np.diag(vals ** -0.5) @ vecs.T
    return normalize_windows(np.einsum("cd,ndt->nct", w, x, optimize=True).astype(np.float32))


def whiten_by_group(x, groups, shrink=0.1):
    out = np.empty_like(x)
    for g in np.unique(groups):
        sel = groups == g
        out[sel] = whiten(x[sel], shrink)
    return out


def offdiag_corr(x):
    cov = channel_cov(x.astype(np.float64))
    d = np.sqrt(np.diag(cov))
    corr = cov / d[:, None] / d[None]
    return float(np.abs(corr[~np.eye(len(corr), dtype=bool)]).mean())
