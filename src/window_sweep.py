"""Which stretch of the epoch should be decoded?

THINGS-EEG2 is RSVP (a new image every 200 ms), so the 1 s epoch holds the
target image and the next four. This sweeps the decoding window [start, end]
after onset with a centered ridge decoder (20 ms bins -> CLIP), choosing
lambda and comparing windows on a validation set carved from the TRAINING
images (10% of them, never used to fit), then reports the test set once per
window for reference.

Retrieval: validation = 200 held-out training images (4 repetitions
averaged); test = the 200 test images (80 repetitions averaged).
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, MODELS, OUTPUTS, SFREQ
from train_encoder import load_features

BIN = 5
LAMBDAS = (1e1, 1e2, 1e3, 1e4)
WINDOWS = [(0, 200), (0, 300), (0, 400), (0, 500), (0, 700), (0, 1000),
           (60, 300), (60, 500), (200, 1000), (500, 1000)]


def featurize(x, start_ms, end_ms):
    a, b = int(start_ms * SFREQ / 1000), int(end_ms * SFREQ / 1000)
    x = x[..., a:b]
    n = x.shape[-1] // BIN * BIN
    x = x[..., :n].reshape(*x.shape[:-1], n // BIN, BIN).mean(-1)
    return x.reshape(*x.shape[:-2], -1).astype(np.float64)


def retrieval(pred, feats):
    sim = F.normalize(torch.from_numpy(pred).float(), dim=-1) @ F.normalize(feats, dim=-1).T
    t = torch.arange(len(pred))
    return (float((sim.argmax(1) == t).float().mean()),
            float((sim.topk(5, 1).indices == t[:, None]).any(1).float().mean()))


def main():
    x_tr = np.load(DATA / "processed/things08_training_21ch.npy", mmap_mode="r")
    x_te = np.load(DATA / "processed/things08_test_21ch.npy", mmap_mode="r")
    f_tr, f_te = load_features("train"), load_features("test")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x_tr))
    val, fit = np.sort(perm[:1654]), np.sort(perm[1654:])
    val_eval = val[:200]
    y = np.repeat(f_tr[fit].numpy().astype(np.float64), x_tr.shape[1], axis=0)
    y_mean = y.mean(0)

    report = {}
    for start, end in WINDOWS:
        z = featurize(np.asarray(x_tr[fit]), start, end).reshape(len(fit) * x_tr.shape[1], -1)
        mu, sd = z.mean(0), z.std(0) + 1e-6
        z = (z - mu) / sd
        zz, zy = z.T @ z, z.T @ (y - y_mean)
        zv = ((featurize(np.asarray(x_tr[val_eval]), start, end) - mu) / sd).mean(1)
        zt = ((featurize(np.asarray(x_te), start, end) - mu) / sd).mean(1)
        best = None
        for lam in LAMBDAS:
            w = np.linalg.solve(zz + lam * np.eye(len(zz)), zy)
            v1, v5 = retrieval(zv @ w, f_tr[val_eval])
            if best is None or v5 + v1 > best["val_top5"] + best["val_top1"]:
                t1, t5 = retrieval(zt @ w, f_te)
                best = {"lambda": lam, "val_top1": v1, "val_top5": v5, "test_top1": t1, "test_top5": t5,
                        "w": w, "mu": mu, "sd": sd}
        key = f"{start}-{end}ms"
        report[key] = {k: v for k, v in best.items() if k not in ("w", "mu", "sd")}
        print(key, json.dumps(report[key]), flush=True)
        torch.save({"w": torch.from_numpy(best["w"]).float(), "mu": torch.from_numpy(best["mu"]).float(),
                    "sd": torch.from_numpy(best["sd"]).float(), "start_ms": start, "end_ms": end},
                   MODELS / f"ridge_window_{start}-{end}ms.pt")
    (OUTPUTS / "window_sweep_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
