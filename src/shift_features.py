"""Time-shift-invariant EEG representations and a ridge decoder to CLIP.

Every feature here is exactly invariant to a circular shift of the 1 s
window (and, for the cross-channel ones, to a shift common to all channels),
so it reads the same thing whether or not the window starts at a stimulus
onset. Used by test_shift_invariance.py to measure how much image information
survives without onset, and by ShiftRobustDecoder to embed arbitrary windows.

Features (input: (n, 21, 250) windows normalized with common.normalize_windows):
  raw        20 ms bins (aligned baseline; not invariant)
  psd        log periodogram per channel, 1-45 Hz
  csd        coherency (Re, Im) between channel pairs plus log auto-spectra,
             in 2 Hz bands up to 31 Hz and one 31-45 Hz band
  bispec     per-channel bispectrum B(f1,f2) = X(f1) X(f2) X*(f1+f2),
             f1+f2 <= 45 Hz, compressed to B/|B|^(2/3), averaged in 3x3 Hz blocks
  scat       two-layer scattering with analytic Morlet filters, modulus and a
             global (circular) time average; log S1 and log S2/S1
"""
import numpy as np
import torch

N = 250
FMAX = 45


def fft(x):
    return np.fft.rfft(x, axis=-1)


def f_raw(x):
    x = x.reshape(*x.shape[:-1], 50, 5).mean(-1)
    return x.reshape(len(x), -1)


def f_psd(x, X=None):
    X = fft(x) if X is None else X
    return np.log(np.abs(X[..., 1:FMAX + 1]) ** 2 + 1e-3).reshape(len(x), -1)


CSD_BANDS = [(f, f + 2) for f in range(1, 31, 2)] + [(31, FMAX + 1)]
_IU = np.triu_indices(21, 1)


def f_csd(x, X=None):
    X = fft(x) if X is None else X
    out = []
    for lo, hi in CSD_BANDS:
        b = X[..., lo:hi]
        s = np.einsum("nif,njf->nij", b, b.conj()) / (hi - lo)
        d = np.sqrt(np.einsum("nii->ni", s).real) + 1e-6
        c = (s / d[:, :, None] / d[:, None, :])[:, _IU[0], _IU[1]]
        out += [c.real, c.imag, np.log(d ** 2)]
    return np.concatenate(out, 1)


_F1, _F2 = np.array([(a, b) for a in range(1, FMAX) for b in range(1, a + 1) if a + b <= FMAX]).T
_BLK = np.unique(np.stack([(_F1 - 1) // 3, (_F2 - 1) // 3], 1), axis=0, return_inverse=True)[1].ravel()
_POOL = np.zeros((len(_F1), _BLK.max() + 1), np.float32)
_POOL[np.arange(len(_F1)), _BLK] = 1
_POOL /= _POOL.sum(0)


def f_bispec(x, X=None):
    X = fft(x) if X is None else X
    B = X[..., _F1] * X[..., _F2] * X[..., _F1 + _F2].conj()
    B = B / (np.abs(B) ** (2 / 3) + 1e-6)
    B = B @ _POOL
    return np.concatenate([B.real, B.imag], -1).reshape(len(x), -1)


def _morlet(xi, q):
    k = np.arange(N, dtype=np.float64)
    sigma = xi * (1 - 2 ** (-1 / q)) * 1.2
    return np.exp(-((k - xi) ** 2) / (2 * max(sigma, 0.6) ** 2)) * (k < N // 2)


XI1 = [FMAX * 2 ** (-j / 4) for j in range(20)]
XI2 = [16, 8, 4, 2, 1]
PSI1 = torch.tensor(np.stack([_morlet(xi, 4) for xi in XI1]), dtype=torch.complex64)
PSI2 = torch.tensor(np.stack([_morlet(xi, 1) for xi in XI2]), dtype=torch.complex64)
PAIRS = [(i, j) for i, a in enumerate(XI1) for j, b in enumerate(XI2) if b < a / 2]


def f_scat(x, X=None):
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    Xf = torch.fft.fft(xt.to(torch.complex64), dim=-1)
    u1 = torch.fft.ifft(Xf[:, :, None] * PSI1, dim=-1).abs()
    s1 = u1.mean(-1)
    U1 = torch.fft.fft(u1.to(torch.complex64), dim=-1)
    i1, i2 = map(list, zip(*PAIRS))
    s2 = torch.fft.ifft(U1[:, :, i1] * PSI2[i2], dim=-1).abs().mean(-1)
    out = torch.cat([torch.log(s1 + 1e-4), torch.log(s2 / (s1[:, :, i1] + 1e-4) + 1e-4)], -1)
    return out.reshape(len(x), -1).numpy()


FEATURES = {"raw": f_raw, "psd": f_psd, "csd": f_csd, "bispec": f_bispec, "scat": f_scat}


def featurize(x, names, chunk=2048):
    parts = []
    for i in range(0, len(x), chunk):
        xc = np.asarray(x[i:i + chunk], dtype=np.float32)
        X = fft(xc)
        parts.append(np.concatenate(
            [FEATURES[n](xc) if n in ("raw", "scat") else FEATURES[n](xc, X) for n in names], 1
        ).astype(np.float32))
    return np.concatenate(parts)


def random_roll(x, rng):
    s = rng.integers(0, x.shape[-1], len(x))
    idx = (np.arange(x.shape[-1])[None] - s[:, None]) % x.shape[-1]
    return np.take_along_axis(x, idx[:, None, :], axis=-1)


class ShiftRobustDecoder:
    """Ridge on concatenated invariant features. Call with (n, 21, 250)
    windows normalized with common.normalize_windows; returns (n, 1024) in
    CLIP space. Retrieval and contrasts should use the output minus self.b."""

    def __init__(self, names, mu, sd, w, b):
        self.names, self.mu, self.sd, self.w, self.b = list(names), mu, sd, w, b

    def __call__(self, x):
        return ((featurize(x, self.names) - self.mu) / self.sd) @ self.w + self.b

    def save(self, path):
        torch.save({"names": self.names, **{k: torch.from_numpy(getattr(self, k)) for k in ("mu", "sd", "w", "b")}},
                   path)

    @classmethod
    def load(cls, path):
        s = torch.load(path)
        return cls(s["names"], *(s[k].numpy() for k in ("mu", "sd", "w", "b")))
