"""Gradient-free variants of a trained ATM-S encoder, derived from reading its
weights (see read_weights.py / weight_surgery.py):

  original  as trained
  detail    the 10 highest-variance temporal filters (the "rhythm" pathway)
            silenced; on THINGS this keeps or improves retrieval
  adabn     BatchNorm statistics recomputed on sober resting EEG of the
            target dataset (no labels, no drug data)
  detail+adabn

Every variant also gets output centering: the mean output on the supplied
baseline windows is subtracted from the final bias, which removes the shared
"default image" (contrasts are unaffected by it).
"""
import torch

from common import MODELS, ChannelTransformerEncoder, device

VARIANTS = ("original", "detail", "adabn", "detail+adabn")
N_RHYTHM_FILTERS = 10


@torch.no_grad()
def build(ckpt_name, variant, baseline_windows):
    dev = device()
    ck = torch.load(MODELS / ckpt_name, map_location="cpu")
    m = ChannelTransformerEncoder(len(ck["channels"]))
    m.load_state_dict(ck["state_dict"])
    m.to(dev).eval()
    if "detail" in variant:
        var = m.tsconv[2].running_var
        for k in torch.argsort(var, descending=True)[:N_RHYTHM_FILTERS]:
            m.tsconv[4].weight[:, k] = 0
    if "adabn" in variant:
        for bn in (m.tsconv[2], m.tsconv[5]):
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()
        for i in range(0, len(baseline_windows), 2048):
            m(torch.from_numpy(baseline_windows[i:i + 2048]).to(dev))
        m.eval()
    out = torch.cat([m(torch.from_numpy(baseline_windows[i:i + 2048]).to(dev))
                     for i in range(0, len(baseline_windows), 2048)])
    m.proj_norm.bias -= out.mean(0)
    return m


@torch.no_grad()
def run(m, x):
    dev = next(m.parameters()).device
    return torch.cat([m(torch.from_numpy(x[i:i + 2048]).to(dev)).cpu() for i in range(0, len(x), 2048)])
