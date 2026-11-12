"""Low-level branch: EEG -> coarse SDXL-VAE latent (layout and colour).

The ATM authors regress the full 4x64x64 latent; EEG carries far less than
16k dimensions, so here the latent is average-pooled to 4xSxS and predicted
with a centered ridge from the 60-500 ms window (see window_sweep.py). The
pooling size and lambda are chosen on validation images carved from the
training set; the test set is reported once.

Metrics: (a) pattern correlation between predicted and true pooled latent
after removing the mean image, against the mean correlation with the other
images; (b) retrieval of the right image among the candidates.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, MODELS, OUTPUTS
from window_sweep import featurize

START, END = 60, 500
SIZES = (4, 8, 16)
LAMBDAS = (1e2, 1e3, 1e4, 1e5)


def load_latents(split):
    d = torch.load(DATA / f"raw/things/{split}_image_latent_512.pt", map_location="cpu", weights_only=False)
    return d["image_latent"].float()


def pooled(lat, size):
    return F.adaptive_avg_pool2d(lat, size).flatten(1).numpy().astype(np.float64)


def metrics(pred, true):
    p = torch.from_numpy(pred - pred.mean(0)).float()
    t = torch.from_numpy(true - true.mean(0)).float()
    sim = F.normalize(p, dim=-1) @ F.normalize(t, dim=-1).T
    n = len(sim)
    idx = torch.arange(n)
    own = sim.diag().mean().item()
    other = ((sim.sum() - sim.diag().sum()) / (n * (n - 1))).item()
    return {"corr_own": own, "corr_other": other,
            "top1": float((sim.argmax(1) == idx).float().mean()),
            "top5": float((sim.topk(5, 1).indices == idx[:, None]).any(1).float().mean())}


def main():
    x_tr = np.load(DATA / "processed/things08_training_21ch.npy", mmap_mode="r")
    x_te = np.load(DATA / "processed/things08_test_21ch.npy", mmap_mode="r")
    lat_tr, lat_te = load_latents("train"), load_latents("test")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x_tr))
    val, fit = np.sort(perm[:1654])[:200], np.sort(perm[1654:])
    reps = x_tr.shape[1]

    z = featurize(np.asarray(x_tr[fit]), START, END).reshape(len(fit) * reps, -1)
    mu, sd = z.mean(0), z.std(0) + 1e-6
    z = (z - mu) / sd
    zz = z.T @ z
    zv = ((featurize(np.asarray(x_tr[val]), START, END) - mu) / sd).mean(1)
    zt = ((featurize(np.asarray(x_te), START, END) - mu) / sd).mean(1)

    report, best = {}, None
    for size in SIZES:
        y = pooled(lat_tr, size)
        y_mean = y[fit].mean(0)
        zy = z.T @ (np.repeat(y[fit], reps, axis=0) - y_mean)
        for lam in LAMBDAS:
            w = np.linalg.solve(zz + lam * np.eye(len(zz)), zy)
            mv = metrics(zv @ w, y[val])
            key = f"{size}x{size}_lam{lam:g}"
            report[key] = {"val": mv}
            print(key, json.dumps({k: round(v, 3) for k, v in mv.items()}), flush=True)
            score = mv["corr_own"] - mv["corr_other"]
            if best is None or score > best[0]:
                best = (score, size, lam, w, y_mean)
    _, size, lam, w, y_mean = best
    mt = metrics(zt @ w, pooled(lat_te, size))
    report["selected"] = {"size": size, "lambda": lam, "test": mt, "window_ms": [START, END]}
    print("selected", size, lam, "test", json.dumps({k: round(v, 3) for k, v in mt.items()}), flush=True)
    torch.save({"w": torch.from_numpy(w).float(), "mu": torch.from_numpy(mu).float(),
                "sd": torch.from_numpy(sd).float(), "y_mean": torch.from_numpy(y_mean).float(),
                "size": size, "start_ms": START, "end_ms": END}, MODELS / "ridge_lowlevel.pt")
    (OUTPUTS / "lowlevel_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
