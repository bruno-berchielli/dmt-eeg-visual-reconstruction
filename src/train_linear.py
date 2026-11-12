"""Linear (ridge) EEG -> CLIP ViT-H/14 decoder on the same 21 shared channels,
in the spirit of Perceptogram. Serves as a low-capacity counterpart to ATM-S.

Usage: python src/train_linear.py --sub 08
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import MODELS, normalize_windows
from train_encoder import load_features, load_split

BIN = 5


def featurize(x):
    """(..., 21, 250) -> (..., 21 * 50): 20 ms bins."""
    x = x.reshape(*x.shape[:-1], x.shape[-1] // BIN, BIN).mean(-1)
    return x.reshape(*x.shape[:-2], -1)


class LinearDecoder:
    def __init__(self, mu, sd, w, b):
        self.mu, self.sd, self.w, self.b = mu, sd, w, b

    def __call__(self, x):
        z = (featurize(x) - self.mu) / self.sd
        return z @ self.w + self.b

    def state(self):
        return {k: torch.from_numpy(getattr(self, k)) for k in ("mu", "sd", "w", "b")}

    @classmethod
    def load(cls, path):
        s = torch.load(path)
        return cls(*(s[k].numpy() for k in ("mu", "sd", "w", "b")))


def fit(z, y, lam):
    a = z.T @ z + lam * np.eye(z.shape[1], dtype=np.float64)
    return np.linalg.solve(a, z.T @ y).astype(np.float32)


def evaluate(dec, x_test, f_test):
    feats = F.normalize(f_test, dim=-1)
    target = torch.arange(len(f_test))
    out = {}
    for name, x in (("avg", normalize_windows(x_test.mean(1))), ("single", x_test[:, 0])):
        sim = F.normalize(torch.from_numpy(dec(x)), dim=-1) @ feats.T
        out[f"top1_{name}"] = (sim.argmax(1) == target).float().mean().item()
        out[f"top5_{name}"] = (sim.topk(5, 1).indices == target[:, None]).any(1).float().mean().item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub", default="08")
    args = p.parse_args()

    x_train = load_split(args.sub, "training")
    x_test = load_split(args.sub, "test")
    f_train = load_features("train").numpy()
    f_test = load_features("test")
    n_cond, n_rep = x_train.shape[:2]

    z = featurize(x_train.reshape(n_cond * n_rep, *x_train.shape[2:])).astype(np.float64)
    y = np.repeat(f_train, n_rep, axis=0).astype(np.float64)
    mu, sd = z.mean(0), z.std(0) + 1e-6
    z = (z - mu) / sd
    b = y.mean(0)
    y = y - b
    zz, zy = z.T @ z, z.T @ y

    best, log = None, []
    for lam in (1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6):
        w = np.linalg.solve(zz + lam * np.eye(len(zz)), zy).astype(np.float32)
        dec = LinearDecoder(mu.astype(np.float32), sd.astype(np.float32), w, b.astype(np.float32))
        m = {"lambda": lam, **evaluate(dec, x_test, f_test), "chance": 1 / len(f_test)}
        log.append(m)
        print(json.dumps(m), flush=True)
        if best is None or m["top5_avg"] > best[0]["top5_avg"]:
            best = (m, dec)
    torch.save(best[1].state(), MODELS / f"linear_sub-{args.sub}.pt")
    (MODELS / f"linear_sub-{args.sub}_log.json").write_text(json.dumps(log, indent=1))
    print("best", json.dumps(best[0]))


if __name__ == "__main__":
    main()
