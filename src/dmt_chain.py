"""Chronological image chain for one DMT recording.

The recording is split into N equal parts. In each part the K highest peaks
of the onset detector (>= 0.5 s apart) are decoded and averaged, the mean at
the surrogate's events is subtracted, and one image is generated:
  chain        image-to-image from the previous part's image (the prior),
               guided by this part's vector
  independent  same vector, no prior (to see what the prior contributes)

Usage: python dmt_chain.py S35 [K=1] [N=20] [strength=0.6]
"""
import sys

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image

from common import DATA, MODELS, OUTPUTS, SFREQ, SHARED_CHANNELS
from dmt_events import decode_at, detector_score
from event_decoders import EventDecoder
from onset_lib import CHANNEL_SETS, CONT, clean_mask, load_cont, segments, valid_starts
from render import as_condition, label, sheet
from sdxl import EmbeddingToImage

GAP = SFREQ // 2


def top_peaks(score, grid, k):
    picked = []
    for i in np.argsort(-score):
        if all(abs(grid[i] - p) >= GAP for p in picked):
            picked.append(int(grid[i]))
            if len(picked) == k:
                break
    return np.array(sorted(picked))


def main():
    subj = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    stem = f"dmt_{subj}_DMT"
    base = torch.load(DATA / "processed/dmt_events_T500multi.pt", weights_only=False)[stem]["surrogate_clip_mean"]
    tpl = np.load(MODELS / "onset_templates_multi.npz")["z14_T500"]
    idx = [SHARED_CHANNELS.index(c) for c in CHANNEL_SETS["z14"]]
    dec = {"clip": EventDecoder.load("clip_multi")}
    x, pieces, _ = load_cont(CONT / f"{stem}.npz", SHARED_CHANNELS)
    grid = valid_starts(segments(clean_mask(x, pieces)), tpl.shape[1])
    grid = grid[grid + SFREQ <= x.shape[1]]
    score = detector_score(x, tpl, idx)[grid]
    edges = np.linspace(0, x.shape[1], n + 1)

    gen = EmbeddingToImage()
    i2i = AutoPipelineForImage2Image.from_pipe(gen.pipe)
    i2i.set_progress_bar_config(disable=True)
    chain, indep, prev = [], [], None
    for j in range(n):
        sel = (grid >= edges[j]) & (grid < edges[j + 1])
        if not sel.any():
            print(f"part {j + 1}: no clean data", flush=True)
            continue
        ev = top_peaks(score[sel], grid[sel], k)
        v = as_condition(torch.from_numpy(decode_at(x, ev, dec)["clip"].mean(0)).float() - base)
        alone = gen(v)
        if prev is None:
            img = alone
        else:
            e = v.reshape(1, 1, -1).to(gen.dev, torch.float16)
            img = i2i(prompt="", image=prev, strength=strength, num_inference_steps=4, guidance_scale=0.0,
                      ip_adapter_image_embeds=[e], generator=torch.Generator("cpu").manual_seed(0)).images[0]
        prev = img
        t = ev[0] / SFREQ
        tag = f"{j + 1}/{n}  {int(t // 60)}:{int(t % 60):02d}"
        chain.append(label(img, tag, 18))
        indep.append(label(alone, tag, 18))
        print(f"part {j + 1}: peaks at {np.round(ev / SFREQ, 1)} s", flush=True)
    out = OUTPUTS / "events"
    rows = lambda ims: [ims[i:i + 5] for i in range(0, len(ims), 5)]
    sheet(rows(chain), size=384).save(out / f"{subj}_chain_k{k}.jpg", quality=90)
    sheet(rows(indep), size=384).save(out / f"{subj}_chain_k{k}_no_prior.jpg", quality=90)


if __name__ == "__main__":
    main()
