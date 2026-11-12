"""Read the trained ATM-S encoder's weights directly (no data, no forward pass).

Along the residual (linear) path, an input x (channels x time) reaches the
temporal conv as x @ W_v^T. Filter k at position d therefore applies an
effective kernel over raw input time:
    g[k, d, t] = sum_j conv[k, j] * W_v[d + j, t]
Its FFT gives that filter's frequency tuning in the raw EEG. The spatial conv
(40 x 40 x n_channels) gives each electrode's weight.

Usage: python src/read_weights.py [--ckpt encoder_sub-08.pt]
"""
import argparse
import json

import numpy as np
import torch

from common import MODELS, OUTPUTS, SFREQ, WINDOW

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def filter_spectra(sd):
    wv = sd["value_embedding.weight"].numpy()
    conv = sd["tsconv.0.weight"].numpy()[:, 0, 0, :]
    k, taps = conv.shape
    positions = wv.shape[0] - taps + 1
    g = np.zeros((k, positions, wv.shape[1]))
    for j in range(taps):
        g += conv[:, None, j, None] * wv[None, j:j + positions, :]
    spec = np.abs(np.fft.rfft(g, axis=-1)).mean(1)
    freqs = np.fft.rfftfreq(WINDOW, 1 / SFREQ)
    band = {b: spec[:, (freqs >= lo) & (freqs < hi)].mean(1) for b, (lo, hi) in BANDS.items()}
    return freqs, spec, band


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="encoder_sub-08.pt")
    args = p.parse_args()
    ck = torch.load(MODELS / args.ckpt, map_location="cpu")
    sd, channels = ck["state_dict"], ck["channels"]

    freqs, spec, band = filter_spectra(sd)
    total = sum(band.values())
    rel = {b: v / total for b, v in band.items()}
    dominant = [max(BANDS, key=lambda b: rel[b][k]) for k in range(len(spec))]

    spatial = sd["tsconv.4.weight"].numpy()[:, :, :, 0]
    ch_weight = np.sqrt((spatial ** 2).sum((0, 1)))
    ch_rank = sorted(zip(channels, ch_weight / ch_weight.max()), key=lambda t: -t[1])

    wv_spec = np.abs(np.fft.rfft(sd["value_embedding.weight"].numpy(), axis=1)).mean(0)
    out = {
        "filter_dominant_band": dominant,
        "filters_per_band": {b: dominant.count(b) for b in BANDS},
        "filter_rel_band_power": {b: [round(float(x), 3) for x in v] for b, v in rel.items()},
        "channel_weight": [(c, round(float(w), 3)) for c, w in ch_rank],
        "value_embedding_band_gain": {b: round(float(wv_spec[(freqs >= lo) & (freqs < hi)].mean()), 4)
                                      for b, (lo, hi) in BANDS.items()},
    }
    name = args.ckpt.replace(".pt", "")
    (OUTPUTS / f"weights_reading_{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "filter_rel_band_power"}, indent=1))


if __name__ == "__main__":
    main()
