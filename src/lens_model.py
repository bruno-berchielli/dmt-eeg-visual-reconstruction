"""Infer the time-window width ("lens") per moment instead of choosing it
from a fixed list: the smallest window whose mean is expected to be
reliable, given how consistent the decoded windows are around that moment.

In the whitened space (dmt_events_video.calibrated_confidence), the local
consistency rho(t) is the mean pairwise cosine among the windows within
PILOT seconds of t. By the Spearman-Brown relation, the mean of n windows
has reliability n*rho / (1 + (n-1)*rho), so the number needed to reach a
target reliability R is n = R(1-rho) / (rho(1-R)). The lens is the Gaussian
width whose effective number of windows equals n at the local window rate
(clamped to [MIN, MAX] seconds; where rho <= 0 nothing is reliable).

Check: split-half agreement actually reached with the inferred lens, next
to the fixed 20 s window. Output: outputs/lens_model.json
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

import dmt_events_video as vid
from common import DATA, OUTPUTS

PILOT, TARGET, MIN, MAX = 30.0, 0.8, 3.0, 60.0


def whitener(d, cond="DMT", n_comp=50):
    pool = torch.cat([r["v"].float() for k, r in d.items() if k.endswith(f"_{cond}")])
    mu = pool.mean(0)
    _, S, Vt = torch.linalg.svd(pool - mu, full_matrices=False)
    P = Vt[:n_comp].T / S[:n_comp] * np.sqrt(len(pool))
    return lambda v: (v - mu) @ P


def infer_lens(t, xw, clock):
    u = F.normalize(xw, dim=1)
    near = (t[None] - clock[:, None]).abs() <= PILOT
    n = near.sum(1).float().clamp(min=2)
    s = (near.float() @ u)
    rho = ((s ** 2).sum(1) - n) / (n * (n - 1))
    rate = n / (2 * PILOT)
    need = TARGET * (1 - rho) / (rho.clamp(min=1e-6) * (1 - TARGET))
    sigma = (need / (2 * np.sqrt(np.pi) * rate)).clamp(MIN, MAX)
    sigma[rho <= 0] = MAX
    k = 7
    pad = torch.cat([sigma[:1].repeat(k), sigma, sigma[-1:].repeat(k)])
    return pad.unfold(0, 2 * k + 1, 1).mean(1), rho


def adaptive_trajectory(t, x, clock, sigma):
    w = torch.exp(-0.5 * ((t[None] - clock[:, None]) / sigma[:, None]) ** 2)
    return (w / w.sum(1, keepdim=True)) @ x


def split_half(t, xw, clock, sigma, n_splits=20, seed=0):
    rng = np.random.default_rng(seed)
    block = (t // 10).long()
    w = torch.exp(-0.5 * ((t[None] - clock[:, None]) / sigma[:, None]) ** 2)
    out = []
    for _ in range(n_splits):
        a = torch.from_numpy(rng.permutation(int(block.max()) + 1) % 2 == 0)[block]
        A = (w[:, a] / w[:, a].sum(1, keepdim=True).clamp(min=1e-9)) @ xw[a]
        B = (w[:, ~a] / w[:, ~a].sum(1, keepdim=True).clamp(min=1e-9)) @ xw[~a]
        out.append(F.cosine_similarity(A, B, dim=1))
    return torch.stack(out).mean(0)


def main():
    d = torch.load(DATA / "processed/dmt_dense.pt", weights_only=False)
    white = whitener(d)
    rep = {}
    for key, r in d.items():
        if not key.endswith("_DMT"):
            continue
        t, xw = r["t"], white(r["v"].float())
        clock = torch.arange(float(t.min()), float(t.max()) + vid.STEP, vid.STEP)
        sigma, rho = infer_lens(t, xw, clock)
        got = split_half(t, xw, clock, sigma)
        fixed = split_half(t, xw, clock, torch.full_like(sigma, 20.0))
        # split-half r of half-size means -> reliability of the full mean (Spearman-Brown)
        rel = lambda c: float((2 * c / (1 + c)).clamp(min=0).mean())
        rep[key.split("_")[1]] = {"rho": float(rho.mean()), "lens_median": float(sigma.median()),
                                  "lens_p10": float(sigma.quantile(0.1)), "lens_p90": float(sigma.quantile(0.9)),
                                  "share_below_15s": float((sigma < 15).float().mean()),
                                  "share_at_max": float((sigma >= MAX - 1).float().mean()),
                                  "reliability_inferred": rel(got), "reliability_fixed20": rel(fixed)}
    (OUTPUTS / "lens_model.json").write_text(json.dumps(rep, indent=1))
    for s in ("S28", "S29", "S04", "S12", "S35"):
        print(s, {k: round(v, 2) for k, v in rep[s].items()})
    g = lambda k: float(np.median([v[k] for v in rep.values()]))
    print("group (median):", {k: round(g(k), 2) for k in ("lens_median", "share_below_15s", "share_at_max", "reliability_inferred", "reliability_fixed20")})


if __name__ == "__main__":
    main()
