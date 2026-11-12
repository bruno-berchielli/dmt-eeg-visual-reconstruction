"""How much image information survives in representations that do not depend
on stimulus onset? Ridge from each representation to CLIP ViT-H/14, THINGS-EEG2
sub-08, 21 shared channels.

Conditions on the test set (200 images x 80 repetitions):
  aligned  epochs as recorded (0-1 s after onset)
  shifted  an independent uniform circular shift (0-249 samples) per trial,
           applied before featurizing; features are averaged across the 80
           repetitions afterwards (never the misaligned waveforms)
Ridge lambda is chosen on the even test images (100-way top-5, averaged
repetitions) and reported on the odd ones (100-way; chance 1% / 5%).
"single" is the mean over all 80 single-trial retrievals. Retrieval uses the
prediction minus the ridge intercept (the mean training CLIP vector), which
otherwise dominates the cosine once predictions are averaged over trials.

Writes outputs/shift_invariance_report.json and models/shift_robust_ridge_sub-08.pt.
Usage: python src/test_shift_invariance.py
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, MODELS, OUTPUTS, normalize_windows
from shift_features import FEATURES, ShiftRobustDecoder, featurize, random_roll
from train_encoder import load_features

LAMBDAS = np.logspace(1, 7, 13)
SEL, EV = np.arange(0, 200, 2), np.arange(1, 200, 2)


def load(split):
    return np.load(DATA / "processed" / f"things08_{split}_21ch.npy", mmap_mode="r")


def retrieval(pred, feats, idx):
    f = F.normalize(feats[idx], dim=-1)
    t = torch.arange(len(idx))
    out = {}
    if pred.ndim == 2:
        sim = F.normalize(torch.from_numpy(pred[idx]).float(), dim=-1) @ f.T
        return {"top1": (sim.argmax(1) == t).float().mean().item(),
                "top5": (sim.topk(5, 1).indices == t[:, None]).any(1).float().mean().item()}
    p = F.normalize(torch.from_numpy(pred[idx]).float(), dim=-1)
    sim = torch.einsum("irk,jk->rij", p, f)
    out["top1"] = (sim.argmax(-1) == t).float().mean().item()
    out["top5"] = (sim.topk(5, -1).indices == t[:, None]).any(-1).float().mean().item()
    return out


def fit_select(z_tr, y, z_te, feats):
    """z_te: (200, 80, d). Returns (report, (mu, sd, w, b))."""
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-6
    zt = torch.from_numpy((z_tr - mu) / sd)
    b = y.mean(0)
    yc = torch.from_numpy(y - b).double()
    zz = torch.zeros(zt.shape[1], zt.shape[1], dtype=torch.float64)
    zy = torch.zeros(zt.shape[1], yc.shape[1], dtype=torch.float64)
    for i in range(0, len(zt), 8192):
        c = zt[i:i + 8192].double()
        zz += c.T @ c
        zy += c.T @ yc[i:i + 8192]
    e, V = torch.linalg.eigh(zz)
    A = V.T @ zy
    te = torch.from_numpy((z_te - mu) / sd).double()
    te_avg_V = te.mean(1) @ V
    sweep = []
    for lam in LAMBDAS:
        pred = (te_avg_V / (e + lam)) @ A
        sweep.append({"lambda": float(lam), **retrieval(pred.numpy(), feats, SEL)})
    best = max(sweep, key=lambda s: (s["top5"], s["top1"]))
    w = (V @ (A / (e + best["lambda"])[:, None])).float()
    pred_trials = (te.float() @ w).numpy()
    pred_avg = pred_trials.mean(1)
    rep = {"lambda": best["lambda"], "dim": int(zt.shape[1]),
           "sel_avg": retrieval(pred_avg, feats, SEL),
           "eval_avg": retrieval(pred_avg, feats, EV),
           "eval_single": retrieval(pred_trials, feats, EV),
           "all200_avg": retrieval(pred_avg, feats, np.arange(200))}
    return rep, (mu.astype(np.float32), sd.astype(np.float32), w.numpy(), b.astype(np.float32))


def main():
    rng = np.random.default_rng(0)
    names = list(FEATURES)
    x_tr = load("training")
    n_cond, n_rep = x_tr.shape[:2]
    flat_tr = x_tr.reshape(n_cond * n_rep, 21, 250)
    x_te = np.asarray(load("test"))
    feats = load_features("test")
    y = np.repeat(load_features("train").numpy(), n_rep, axis=0).astype(np.float64)

    cache = DATA / "processed" / "things08_training_shiftfeat.npy"
    if not cache.exists():
        np.save(cache, featurize(flat_tr, names))
    z_tr = np.load(cache)
    z_tr_raw_shift = np.concatenate([featurize(random_roll(np.asarray(flat_tr[i:i + 8192]), rng), ["raw"])
                                     for i in range(0, len(flat_tr), 8192)])
    te_flat = x_te.reshape(-1, 21, 250)
    z_te = {"aligned": featurize(te_flat, names).reshape(200, 80, -1),
            "shifted": featurize(random_roll(te_flat, rng), names).reshape(200, 80, -1)}
    dims = {n: FEATURES[n](te_flat[:2]).shape[1] if n in ("raw", "scat") else
            FEATURES[n](te_flat[:2], np.fft.rfft(te_flat[:2], axis=-1)).shape[1] for n in names}
    off = np.cumsum([0] + [dims[n] for n in names])
    cols = {n: slice(off[i], off[i + 1]) for i, n in enumerate(names)}

    def take(z, ns):
        return np.concatenate([z[..., cols[n]] for n in ns], -1)

    report = {"protocol": "100-way on odd test images (chance top1 1%, top5 5%); lambda chosen on even images",
              "results": {}}

    def run(label, ns):
        zt = take(z_tr, ns)
        out = {}
        for cond in ("aligned", "shifted"):
            r, params = fit_select(zt, y, take(z_te[cond], ns), feats)
            out[cond] = r
            print(label, cond, json.dumps({k: v for k, v in r.items() if k in ("lambda", "eval_avg", "eval_single")}),
                  flush=True)
        report["results"][label] = out
        return params

    run("raw (ridge, trained aligned)", ["raw"])
    out = {}
    for cond in ("aligned", "shifted"):
        r, _ = fit_select(z_tr_raw_shift, y, take(z_te[cond], ["raw"]), feats)
        out[cond] = r
        print("raw shift-trained", cond, json.dumps(r["eval_avg"]), flush=True)
    report["results"]["raw (ridge, trained shifted)"] = out
    del z_tr_raw_shift

    wave_avg = featurize(normalize_windows(x_te.mean(1)), ["raw"])
    r, _ = fit_select(take(z_tr, ["raw"]), y, wave_avg[:, None], feats)
    report["results"]["raw (ridge, waveform averaged over reps, as train_linear.py)"] = {"aligned": r}
    print("raw waveform-avg", json.dumps(r["eval_avg"]), flush=True)

    for n in ("psd", "csd", "bispec", "scat"):
        run(n, [n])
    run("psd+scat", ["psd", "scat"])
    params = run("psd+csd+bispec+scat", ["psd", "csd", "bispec", "scat"])

    res = report["results"]
    inv = [k for k in res if k.split(" ")[0] in ("psd", "csd", "bispec", "scat", "psd+scat", "psd+csd+bispec+scat")]
    best = max(inv, key=lambda k: (res[k]["shifted"]["sel_avg"]["top5"], res[k]["shifted"]["sel_avg"]["top1"]))
    report["best_invariant_by_selection_half"] = best
    if best != "psd+csd+bispec+scat":
        params = fit_select(take(z_tr, best.split("+")), y, take(z_te["shifted"], best.split("+")), feats)[1]
    dec = ShiftRobustDecoder(best.split("+"), *params)
    dec.save(MODELS / "shift_robust_ridge_sub-08.pt")
    (OUTPUTS / "shift_invariance_report.json").write_text(json.dumps(report, indent=1))
    print("best", best)


if __name__ == "__main__":
    main()
