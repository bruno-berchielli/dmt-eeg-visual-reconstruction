"""Learned shift-robust EEG -> CLIP encoders on THINGS-EEG2 sub-08, 21 shared
channels, same loss and optimizer as train_encoder.py.

  --arch atm  ChannelTransformerEncoder trained with an independent uniform
              circular shift (0-249 samples) per training trial
  --arch cnn  ShiftInvariantCNN: dilated temporal convolutions with circular
              padding and stride 1, then mean and max over time, so the
              embedding is exactly invariant to circular shifts

Epoch chosen on the even test images (100-way top-5, randomly shifted trials,
embeddings averaged over repetitions), reported on the odd ones. Embeddings
are centered on the mean output for 4096 shifted training trials before
retrieval (as the "center" edit in weight_surgery.py). Circular
shifts of RSVP epochs keep two onset cues that onset-free data lack: the
wrap seam and the 5 Hz phase of the stimulus train, so these numbers are an
upper bound for windows cut at arbitrary times.

Usage: python src/train_shift_robust.py --arch cnn --epochs 30
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, MODELS, OUTPUTS, ChannelTransformerEncoder, clip_loss, device, normalize_windows
from train_encoder import load_features

SEL, EV = np.arange(0, 200, 2), np.arange(1, 200, 2)


class ShiftInvariantCNN(nn.Module):
    def __init__(self, n_channels=21, width=128, emb_dim=1024):
        super().__init__()
        layers, c_in = [], n_channels
        for c_out, dil in ((64, 1), (width, 2), (width, 4), (2 * width, 8)):
            layers += [nn.Conv1d(c_in, c_out, 13, dilation=dil, padding=6 * dil, padding_mode="circular"),
                       nn.BatchNorm1d(c_out), nn.GELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.proj_in = nn.Linear(4 * width, emb_dim)
        self.proj_res = nn.Sequential(nn.GELU(), nn.Linear(emb_dim, emb_dim), nn.Dropout(0.5))
        self.proj_norm = nn.LayerNorm(emb_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, x):
        h = self.conv(x)
        h = self.proj_in(self.drop(torch.cat([h.mean(-1), h.amax(-1)], 1)))
        return self.proj_norm(h + self.proj_res(h))


def roll(x, s):
    idx = (torch.arange(x.shape[-1], device=x.device)[None] - s[:, None]) % x.shape[-1]
    return torch.gather(x, -1, idx[:, None, :].expand_as(x))


@torch.no_grad()
def embed(model, x, dev, bs=2048):
    return torch.cat([model(x[i:i + bs].to(dev)).cpu() for i in range(0, len(x), bs)])


def retrieval(emb, feats, idx):
    f = F.normalize(feats[idx], dim=-1)
    t = torch.arange(len(idx))
    p = F.normalize(emb[idx], dim=-1)
    if p.ndim == 2:
        p = p[:, None]
    sim = torch.einsum("irk,jk->rij", p, f)
    return {"top1": (sim.argmax(-1) == t).float().mean().item(),
            "top5": (sim.topk(5, -1).indices == t[:, None]).any(-1).float().mean().item()}


@torch.no_grad()
def evaluate(model, tests, feats, dev, center):
    model.eval()
    mean_out = embed(model, center, dev).mean(0)
    out = {}
    for cond, x in tests.items():
        e = embed(model, x, dev) - mean_out
        if cond == "aligned_waveform_avg":
            out[cond] = {"sel": retrieval(e, feats, SEL), "eval": retrieval(e, feats, EV)}
        else:
            e = e.reshape(200, 80, -1)
            out[cond] = {"sel": retrieval(e.mean(1), feats, SEL), "eval": retrieval(e.mean(1), feats, EV),
                         "eval_single": retrieval(e, feats, EV)}
    model.train()
    return out


def test_sets(seed=0):
    x = np.load(DATA / "processed" / "things08_test_21ch.npy")
    flat = torch.from_numpy(x.reshape(-1, 21, 250))
    s = torch.from_numpy(np.random.default_rng(seed).integers(0, 250, len(flat)))
    return {"aligned": flat, "shifted": roll(flat, s),
            "aligned_waveform_avg": torch.from_numpy(normalize_windows(x.mean(1)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("atm", "cnn"), default="cnn")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    dev = device()
    torch.manual_seed(0)
    x_tr = np.load(DATA / "processed" / "things08_training_21ch.npy")
    n_cond, n_rep = x_tr.shape[:2]
    x_flat = torch.from_numpy(x_tr.reshape(-1, 21, 250)).to(dev)
    del x_tr
    y_idx = torch.arange(n_cond, device=dev).repeat_interleave(n_rep)
    f_train = load_features("train").to(dev)
    feats = load_features("test")
    tests = test_sets()
    g = torch.Generator().manual_seed(1)
    pick = torch.randperm(len(x_flat), generator=g)[:4096].to(dev)
    center = roll(x_flat[pick], torch.randint(0, 250, (4096,), generator=g).to(dev)).cpu()

    model = (ChannelTransformerEncoder(21) if args.arch == "atm" else ShiftInvariantCNN()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    log, best = [], None
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(x_flat), device=dev)
        total = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb = roll(x_flat[b], torch.randint(0, 250, (len(b),), device=dev))
            fb = f_train[y_idx[b]]
            out = model(xb)
            loss = 0.9 * F.mse_loss(out, fb) * 10 + 0.1 * clip_loss(out, fb, model.logit_scale) * 10
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        m = evaluate(model, tests, feats, dev, center)
        m.update(epoch=epoch, loss=total / len(perm), secs=round(time.time() - t0, 1))
        log.append(m)
        print(json.dumps({"epoch": epoch, "loss": round(m["loss"], 3), "secs": m["secs"],
                          **{c: m[c]["eval"] for c in ("aligned", "shifted")},
                          "shifted_sel": m["shifted"]["sel"]}), flush=True)
        key = (m["shifted"]["sel"]["top5"], m["shifted"]["sel"]["top1"])
        if best is None or key > best[0]:
            best = (key, m)
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "metrics": m},
                       MODELS / f"shift_robust_{args.arch}_sub-08.pt")
    (OUTPUTS / f"shift_robust_{args.arch}_report.json").write_text(
        json.dumps({"best": best[1], "log": log}, indent=1))
    print("best", json.dumps(best[1]))


if __name__ == "__main__":
    main()
