"""Jitter-tolerant decoders for event-locked windows.

Centered ridges trained on THINGS (21 channels shared with the DMT montage),
waveform in 20 ms bins from 100 to 540 ms after onset, on copies of every
trial shifted by -40..+40 ms, so that an onset error of that size does not
destroy decoding (see tolerance_test.py):
  clip      -> CLIP ViT-H/14 image embedding (content)
  lowlevel  -> SDXL-VAE latent pooled to 4x4 (layout and colour)
Hyperparameters come from tolerance_test.py / lowlevel.py; all training
images are used. Test-set numbers (aligned and jittered) are reported.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, MODELS, OUTPUTS
from lowlevel import load_latents, metrics, pooled
from tolerance_test import A, B, bins, retrieval
from train_encoder import load_features

OFFSETS = (-10, -5, 0, 5, 10)
LAMBDA = {"clip": 1e4, "lowlevel": 1e4}
SIZE = 4


def features(x, off=0):
    """x: (n, 21, 250) normalized windows starting at the assumed onset."""
    return bins(x, A + off, B + off, 5).reshape(len(x), -1)


class EventDecoder:
    def __init__(self, ck):
        self.w, self.mu, self.sd = (ck[k].numpy() for k in ("w", "mu", "sd"))
        self.y_mean = ck["y_mean"].numpy()

    @classmethod
    def load(cls, name):
        return cls(torch.load(MODELS / f"event_ridge_{name}.pt"))

    def __call__(self, x):
        """Centered prediction (add y_mean to get the absolute target)."""
        return ((features(x) - self.mu) / self.sd) @ self.w


def main():
    x_tr = np.load(DATA / "processed/things08_training_21ch.npy", mmap_mode="r")
    x_te = np.asarray(np.load(DATA / "processed/things08_test_21ch.npy", mmap_mode="r")).reshape(-1, 21, 250)
    reps = x_tr.shape[1]
    xf = np.asarray(x_tr).reshape(-1, 21, 250)
    z = np.concatenate([features(xf, o) for o in OFFSETS]).astype(np.float64)
    mu, sd = z.mean(0), z.std(0) + 1e-6
    z = (z - mu) / sd
    zz = z.T @ z
    targets = {"clip": load_features("train").numpy().astype(np.float64),
               "lowlevel": pooled(load_latents("train"), SIZE)}
    tests = {"clip": load_features("test"), "lowlevel": pooled(load_latents("test"), SIZE)}
    rng = np.random.default_rng(0)
    jit = rng.integers(-10, 11, len(x_te))
    zt0 = ((features(x_te) - mu) / sd).reshape(200, 80, -1).mean(1)
    ztj = np.empty((len(x_te), len(mu)), np.float32)
    for o in np.unique(jit):
        ztj[jit == o] = (features(x_te[jit == o], int(o)) - mu) / sd
    ztj = ztj.reshape(200, 80, -1).mean(1)
    report = {}
    for name, y in targets.items():
        y_mean = y.mean(0)
        yy = np.tile(np.repeat(y - y_mean, reps, axis=0), (len(OFFSETS), 1))
        w = np.linalg.solve(zz + LAMBDA[name] * np.eye(len(zz)), z.T @ yy)
        torch.save({"w": torch.from_numpy(w).float(), "mu": torch.from_numpy(mu).float(),
                    "sd": torch.from_numpy(sd).float(), "y_mean": torch.from_numpy(y_mean).float()},
                   MODELS / f"event_ridge_{name}.pt")
        if name == "clip":
            report[name] = {"aligned": retrieval(zt0 @ w, tests[name]), "jitter_pm40ms": retrieval(ztj @ w, tests[name])}
        else:
            report[name] = {"aligned": metrics(zt0 @ w, tests[name]), "jitter_pm40ms": metrics(ztj @ w, tests[name])}
        print(name, json.dumps(report[name]), flush=True)
    (OUTPUTS / "event_decoders_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
