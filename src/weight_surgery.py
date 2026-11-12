"""Gradient-free edits to the trained ATM-S encoder, chosen on one half of the
THINGS test images and evaluated on the other half (100-way each).

Edits:
  center   shift the output LayerNorm bias by minus the mean output on
           unlabeled THINGS training EEG (removes the shared "default image")
  prune    greedily silence temporal filters (zero their spatial-conv
           input weights) when doing so improves the selection half
Writes the edited checkpoint as encoder_sub-08_edited.pt.
"""
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import MODELS, OUTPUTS, SHARED_CHANNELS, ChannelTransformerEncoder, device, normalize_windows
from train_encoder import load_features, load_split


@torch.no_grad()
def run(model, x, dev):
    return torch.cat([model(torch.from_numpy(x[i:i + 2048]).to(dev)).cpu() for i in range(0, len(x), 2048)])


@torch.no_grad()
def score(model, x_avg, x_single, feats, idx, dev):
    f = F.normalize(feats[idx], dim=-1)
    t = torch.arange(len(idx))
    out = {}
    for name, x in (("avg", x_avg), ("single", x_single)):
        sim = F.normalize(run(model, x[idx], dev), dim=-1) @ f.T
        out[f"top1_{name}"] = (sim.argmax(1) == t).float().mean().item()
        out[f"top5_{name}"] = (sim.topk(5, 1).indices == t[:, None]).any(1).float().mean().item()
    return out


@torch.no_grad()
def main():
    dev = device()
    ck = torch.load(MODELS / "encoder_sub-08.pt", map_location="cpu")
    base = ChannelTransformerEncoder(len(SHARED_CHANNELS))
    base.load_state_dict(ck["state_dict"])
    base.to(dev).eval()

    x_test = load_split("08", "test")
    feats = load_features("test")
    x_avg = normalize_windows(x_test.mean(1))
    x_single = np.ascontiguousarray(x_test[:, 0])
    sel, ev = np.arange(0, 200, 2), np.arange(1, 200, 2)

    x_train = load_split("08", "training")
    rng = np.random.default_rng(0)
    pick = rng.choice(x_train.shape[0], 5000, replace=False)
    unlabeled = np.ascontiguousarray(x_train[pick, 0])
    del x_train

    log = {"baseline": {"sel": score(base, x_avg, x_single, feats, sel, dev),
                        "eval": score(base, x_avg, x_single, feats, ev, dev)}}

    centered = copy.deepcopy(base)
    mean_out = run(centered, unlabeled, dev).mean(0).to(dev)
    centered.proj_norm.bias -= mean_out
    log["center"] = {"sel": score(centered, x_avg, x_single, feats, sel, dev),
                     "eval": score(centered, x_avg, x_single, feats, ev, dev)}

    def metric(m):
        s = score(m, x_avg, x_single, feats, sel, dev)
        return s["top5_avg"] + s["top1_avg"] + 0.5 * (s["top5_single"] + s["top1_single"])

    model = copy.deepcopy(centered)
    best = metric(model)
    removed = []
    for k in range(40):
        trial = copy.deepcopy(model)
        trial.tsconv[4].weight[:, k] = 0
        m = metric(trial)
        if m > best + 1e-6:
            model, best = trial, m
            removed.append(k)
    log["center+prune"] = {"removed_filters": removed,
                           "sel": score(model, x_avg, x_single, feats, sel, dev),
                           "eval": score(model, x_avg, x_single, feats, ev, dev)}

    ck_out = dict(ck)
    ck_out["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    ck_out["edits"] = {"center": True, "removed_filters": removed}
    torch.save(ck_out, MODELS / "encoder_sub-08_edited.pt")
    (OUTPUTS / "weight_surgery_report.json").write_text(json.dumps(log, indent=1))
    for k, v in log.items():
        print(k, {s: {m: round(x, 3) for m, x in d.items()} for s, d in v.items() if s in ("sel", "eval")},
              v.get("removed_filters", ""))


if __name__ == "__main__":
    main()
