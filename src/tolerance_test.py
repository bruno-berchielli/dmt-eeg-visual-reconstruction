"""Which input transformation makes aligned decoding tolerant to small onset
errors while keeping the content?

All decoders are centered ridges to CLIP trained on THINGS (21 channels),
nominal window 100-540 ms after onset, compared on the 200 test images
(80 repetitions averaged) when the assumed onset is off by a fixed amount
or jittered independently per trial. Lambda is chosen on validation images
carved from the training set, at zero offset.

  wave20      waveform in 20 ms bins (baseline)
  wave60      waveform in 60 ms bins (temporal smoothing)
  wave100     waveform in 100 ms bins
  logenv      log band envelopes (delta..gamma) in 60 ms bins
  wave20+env  both
  wave20_jit  wave20 trained on copies shifted by -40..+40 ms
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, OUTPUTS, SFREQ
from train_encoder import load_features

A, B = 25, 135
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
OFFSETS_MS = (-40, -20, 0, 20, 40, 100)
LAMBDAS = (1e2, 1e3, 1e4)


def envelopes(x):
    """x: (..., T) -> (..., n_bands, T) analytic amplitude per band."""
    T = x.shape[-1]
    f = np.fft.fftfreq(T, 1 / SFREQ)
    spec = np.fft.fft(x, axis=-1)
    out = []
    for lo, hi in BANDS:
        h = np.zeros(T)
        h[(f >= lo) & (f < hi)] = 2.0
        out.append(np.abs(np.fft.ifft(spec * h, axis=-1)))
    return np.stack(out, -2).astype(np.float32)


def bins(x, a, b, width):
    x = x[..., a:b]
    n = x.shape[-1] // width * width
    return x[..., :n].reshape(*x.shape[:-1], n // width, width).mean(-1)


def make_features(kind):
    def wave(x, off, width):
        return bins(x, A + off, B + off, width).reshape(len(x), -1)

    def env(x, off):
        return np.log(bins(envelopes(x), A + off, B + off, 15) + 1e-3).reshape(len(x), -1)

    return {"wave20": lambda x, o: wave(x, o, 5), "wave60": lambda x, o: wave(x, o, 15),
            "wave100": lambda x, o: wave(x, o, 25), "logenv": env,
            "wave20+env": lambda x, o: np.concatenate([wave(x, o, 5), env(x, o)], 1),
            "wave20_jit": lambda x, o: wave(x, o, 5)}[kind]


def retrieval(pred, feats):
    sim = F.normalize(torch.from_numpy(pred).float(), dim=-1) @ F.normalize(feats, dim=-1).T
    t = torch.arange(len(pred))
    return [round(float((sim.argmax(1) == t).float().mean()), 3),
            round(float((sim.topk(5, 1).indices == t[:, None]).any(1).float().mean()), 3)]


def main():
    x_tr = np.load(DATA / "processed/things08_training_21ch.npy", mmap_mode="r")
    x_te = np.asarray(np.load(DATA / "processed/things08_test_21ch.npy", mmap_mode="r"))
    f_tr, f_te = load_features("train"), load_features("test")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x_tr))
    val, fit = np.sort(perm[:1654])[:200], np.sort(perm[1654:])
    reps = x_tr.shape[1]
    xf = np.asarray(x_tr[fit]).reshape(len(fit) * reps, 21, 250)
    xv = np.asarray(x_tr[val]).reshape(200 * reps, 21, 250)
    xt = x_te.reshape(200 * 80, 21, 250)
    y = np.repeat(f_tr[fit].numpy().astype(np.float64), reps, axis=0)
    y_mean = y.mean(0)
    report = {}
    for kind in ("wave20", "wave60", "wave100", "logenv", "wave20+env", "wave20_jit"):
        feat = make_features(kind)
        if kind == "wave20_jit":
            offs = (-10, -5, 0, 5, 10)
            z = np.concatenate([feat(xf, o) for o in offs]).astype(np.float64)
            yy = np.tile(y, (len(offs), 1))
        else:
            z, yy = feat(xf, 0).astype(np.float64), y
        mu, sd = z.mean(0), z.std(0) + 1e-6
        z = (z - mu) / sd
        zz, zy = z.T @ z, z.T @ (yy - y_mean)
        del z
        zv = ((feat(xv, 0) - mu) / sd).reshape(200, reps, -1).mean(1)
        best = None
        for lam in LAMBDAS:
            w = np.linalg.solve(zz + lam * np.eye(len(zz)), zy)
            v = retrieval(zv @ w, f_tr[val])
            if best is None or sum(v) > sum(best[1]):
                best = (lam, v, w)
        lam, v, w = best
        res = {"lambda": lam, "val": v, "n_features": int(len(mu))}
        for off in OFFSETS_MS:
            zt = ((feat(xt, int(off * SFREQ / 1000)) - mu) / sd).reshape(200, 80, -1).mean(1)
            res[f"offset_{off}ms"] = retrieval(zt @ w, f_te)
        jit = rng.integers(-10, 11, len(xt))
        zt = np.empty((len(xt), len(mu)), np.float32)
        for o in np.unique(jit):
            zt[jit == o] = (feat(xt[jit == o], int(o)) - mu) / sd
        res["jitter_pm40ms"] = retrieval(zt.reshape(200, 80, -1).mean(1) @ w, f_te)
        report[kind] = res
        print(kind, json.dumps(res), flush=True)
    (OUTPUTS / "tolerance_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
