"""Event-locked decoding and images for ONE subject without the inclusion
rules of dmt_events.py / dmt_events_render.py (>= 1 min of clean data,
>= 10 events). For looking at excluded recordings; a handful of events
carries very little.

Usage: python dmt_events_one.py S27
Output: outputs/events/<subject>_forced.jpg and the per-condition PNGs.
"""
import sys
import zlib

import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

from common import MODELS, OUTPUTS, SHARED_CHANNELS
from dmt_events import QUANTILE, decode_at, detector_score
from dmt_events_render import LowLevel
from event_decoders import EventDecoder
from onset_lib import CHANNEL_SETS, CONT, clean_mask, count_events, load_cont, segments, surrogate, valid_starts
from render import as_condition, label, sheet
from sdxl import EmbeddingToImage

KINDS = {"T500multi": ("onset_templates_multi.npz", "z14_T500", "clip_multi"),
         "T200": ("onset_templates.npz", "z14_T200", "clip")}


def main():
    subj = sys.argv[1]
    idx = [SHARED_CHANNELS.index(c) for c in CHANNEL_SETS["z14"]]
    gen = EmbeddingToImage()
    low = LowLevel(gen.dev)
    i2i = AutoPipelineForImage2Image.from_pipe(gen.pipe)
    i2i.set_progress_bar_config(disable=True)
    out = OUTPUTS / "events"
    rows = []
    for kind, (tpl_file, tpl_key, clip_name) in KINDS.items():
        tpl = np.load(MODELS / tpl_file)[tpl_key]
        decoders = {"clip": EventDecoder.load(clip_name), "lowlevel": EventDecoder.load("lowlevel")}
        for cond in ("DMT", "EC", "EO"):
            path = CONT / f"dmt_{subj}_{cond}.npz"
            if not path.exists():
                continue
            x, pieces, _ = load_cont(path, SHARED_CHANNELS)
            segs = segments(clean_mask(x, pieces))
            grid = valid_starts(segs, tpl.shape[1])
            rng = np.random.default_rng(zlib.crc32(path.stem.encode()))
            s1, s2 = surrogate(x, segs, rng), surrogate(x, segs, rng)
            thr = float(np.quantile(detector_score(s1, tpl, idx)[grid], QUANTILE))
            ev = count_events(detector_score(x, tpl, idx)[grid], grid, segs, thr)
            ev_s = count_events(detector_score(s2, tpl, idx)[grid], grid, segs, thr)
            print(f"{kind} {cond}: {len(grid) * 5 / 250 / 60:.2f} min, {len(ev)} events, surrogate {len(ev_s)}", flush=True)
            if len(ev) == 0:
                continue
            e = decode_at(x, ev, decoders)
            # with no surrogate events, fall back to random clean times as the baseline
            base = decode_at(s2, ev_s, decoders) if len(ev_s) else decode_at(x, rng.choice(grid, min(500, len(grid)), replace=False), decoders)
            vc = torch.from_numpy(e["clip"].mean(0) - base["clip"].mean(0)).float()
            vl = torch.from_numpy(e["lowlevel"].mean(0) - base["lowlevel"].mean(0)).float()
            li = low.image(vl)
            cond_e = as_condition(vc).reshape(1, 1, -1).to(gen.dev, torch.float16)
            mixed = i2i(prompt="", image=li, strength=0.75, num_inference_steps=4, guidance_scale=0.0,
                        ip_adapter_image_embeds=[cond_e], generator=torch.Generator("cpu").manual_seed(0)).images[0]
            imgs = [gen(as_condition(vc)), li.resize((512, 512), Image.BICUBIC), mixed]
            row = []
            for im, t in zip(imgs, ("content", "color-layout", "mixed")):
                im.save(out / f"{subj}_{cond}_{t}_{kind}.png")
                row.append(label(im, f"{subj} {cond} {t} {kind} ({len(ev)} ev)", 16))
            rows.append(row)
    sheet(rows, size=384).save(out / f"{subj}_forced.jpg", quality=90)


if __name__ == "__main__":
    main()
