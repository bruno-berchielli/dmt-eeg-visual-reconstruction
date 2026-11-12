"""Train an ATM-S style EEG -> CLIP ViT-H/14 encoder on THINGS-EEG2 restricted
to the channels available in a target dataset (default: the 21 shared with the
Zenodo DMT montage).

Usage: python src/train_encoder.py --sub 08 --epochs 40 [--channels Oz,Pz --tag _osf]
"""
import argparse
import json
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (DATA, MODELS, SHARED_CHANNELS, THINGS_CHANNELS,
                    ChannelTransformerEncoder, clip_loss, device,
                    normalize_windows)


class NumpyOnlyUnpickler(pickle.Unpickler):
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module, name)


def load_split(sub, split, channels=SHARED_CHANNELS):
    path = DATA / "raw/things" / f"sub-{sub}" / f"preprocessed_eeg_{split}.npy"
    with open(path, "rb") as f:
        arr = NumpyOnlyUnpickler(f).load()["preprocessed_eeg_data"]
    idx = [THINGS_CHANNELS.index(c) for c in channels]
    return normalize_windows(arr[:, :, idx, :].astype(np.float32))


def load_features(split):
    feats = torch.load(DATA / "raw/things" / f"ViT-H-14_features_{split}.pt", map_location="cpu")
    if isinstance(feats, dict):
        feats = feats.get("img_features", next(iter(feats.values())))
    return feats.float()


@torch.no_grad()
def evaluate(model, x_test, f_test, dev):
    model.eval()
    feats_n = F.normalize(f_test.to(dev), dim=-1)
    avg = torch.from_numpy(normalize_windows(x_test.mean(axis=1))).to(dev)
    emb = F.normalize(model(avg), dim=-1)
    sim = emb @ feats_n.T
    target = torch.arange(len(f_test), device=dev)
    top1 = (sim.argmax(1) == target).float().mean().item()
    top5 = (sim.topk(5, dim=1).indices == target[:, None]).any(1).float().mean().item()
    single = torch.from_numpy(x_test[:, 0]).to(dev)
    s1 = (F.normalize(model(single), dim=-1) @ feats_n.T).argmax(1)
    single_top1 = (s1 == target).float().mean().item()
    model.train()
    return {"top1_avg": top1, "top5_avg": top5, "top1_single": single_top1, "chance": 1 / len(f_test)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub", default="08")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--channels", default=",".join(SHARED_CHANNELS))
    p.add_argument("--tag", default="")
    args = p.parse_args()
    channels = args.channels.split(",")

    dev = device()
    torch.manual_seed(0)
    x_train = load_split(args.sub, "training", channels)
    x_test = load_split(args.sub, "test", channels)
    f_train = load_features("train")
    f_test = load_features("test")
    n_cond, n_rep = x_train.shape[:2]
    print(f"train {x_train.shape} test {x_test.shape} feats {tuple(f_train.shape)}", flush=True)

    x_flat = torch.from_numpy(x_train.reshape(n_cond * n_rep, *x_train.shape[2:]))
    y_idx = torch.arange(n_cond).repeat_interleave(n_rep)
    f_train_dev = f_train.to(dev)

    model = ChannelTransformerEncoder(len(channels)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    alpha = 0.9
    log = []
    best = -1
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(x_flat))
        total = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb = x_flat[b].to(dev)
            fb = f_train_dev[y_idx[b].to(dev)]
            out = model(xb)
            loss = alpha * F.mse_loss(out, fb) * 10 + (1 - alpha) * clip_loss(out, fb, model.logit_scale) * 10
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        metrics = evaluate(model, x_test, f_test, dev)
        metrics.update(epoch=epoch, loss=total / len(perm), secs=round(time.time() - t0, 1))
        log.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["top5_avg"] > best:
            best = metrics["top5_avg"]
            torch.save({"state_dict": model.state_dict(), "channels": channels,
                        "metrics": metrics, "sub": args.sub},
                       MODELS / f"encoder_sub-{args.sub}{args.tag}.pt")
    (MODELS / f"encoder_sub-{args.sub}{args.tag}_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
