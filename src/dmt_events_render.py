"""Statistics and images for the event-locked DMT decoding (dmt_events.py).

Per recording, the event-locked component is
    mean decoded vector at detected events - mean at the surrogate's events
(the second term is what detector + decoder return for structureless data
with the same spectrum). Reported: event rates, across-subject consistency
of that component per condition and of DMT minus eyes closed, nearest
THINGS concepts. Rendered per condition and per subject:
  content    CLIP vector -> IP-Adapter + SDXL-Turbo (direct and pulled to
             the nearest real THINGS concepts)
  low-level  predicted 4x4 latent, rescaled to the typical size of a real
             image's deviation from the mean image, upsampled and decoded
             by the SDXL VAE (colour / layout only)
  mixed      SDXL-Turbo image-to-image from the low-level picture, guided by
             the content vector
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoPipelineForImage2Image
from PIL import Image

from common import DATA, MODELS, OUTPUTS
from decode import ConceptSpace, sign_flip_null
from lowlevel import load_latents
from render import Concepts, as_condition, label, load_scales, sheet
from sdxl import EmbeddingToImage

SIZE = 4


class LowLevel:
    def __init__(self, dev):
        self.dev = dev
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae", variant="fp16",
                                                 torch_dtype=torch.float32).to(dev)
        lat = F.adaptive_avg_pool2d(load_latents("test"), SIZE).flatten(1)
        self.typical_norm = (lat - lat.mean(0)).norm(dim=1).median().item()
        self.y_mean = torch.load(MODELS / "event_ridge_lowlevel.pt")["y_mean"]

    @torch.no_grad()
    def image(self, v):
        lat = self.y_mean + F.normalize(v, dim=0) * self.typical_norm
        lat = F.interpolate(lat.reshape(1, 4, SIZE, SIZE), size=64, mode="bicubic", align_corners=False)
        img = self.vae.decode(lat.to(self.dev)).sample[0].clamp(-1, 1).cpu()
        return Image.fromarray(((img.permute(1, 2, 0) + 1) * 127.5).numpy().astype(np.uint8))


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "T200"
    d = torch.load(DATA / f"processed/dmt_events_{kind}.pt", weights_only=False)
    rows = []
    for v in d.values():
        if v["events_clip_mean"] is None or v["surrogate_clip_mean"] is None or len(v["events_s"]) < 10:
            continue
        rows.append({"s": v["subject"], "c": v["condition"], "n": len(v["events_s"]),
                     "rate": v["rate"], "surr": v["rate_surrogate"],
                     "clip": v["events_clip_mean"] - v["surrogate_clip_mean"],
                     "low": v["events_lowlevel_mean"] - v["surrogate_lowlevel_mean"]})
    df = pd.DataFrame(rows)
    cs = ConceptSpace()
    report = {"detector": kind, "rates": {}, "content": {}}
    for c, g in df.groupby("c"):
        report["rates"][c] = {"n": len(g), "events_per_min": float(g.rate.mean()),
                              "surrogate_per_min": float(g.surr.mean()), "events_per_recording": float(g.n.mean())}
    group = {}
    for c, g in df.groupby("c"):
        for space in ("clip", "low"):
            m = torch.stack(list(g[space]))
            obs, null, p = sign_flip_null(m)
            report["content"][f"{c}_{space}"] = {"n": len(m), "consistency": obs, "p": p}
            group[(c, space)] = m.mean(0)
        report["content"][f"{c}_clip"]["top"] = cs.top(group[(c, "clip")], 10)
    piv = df.pivot(index="s", columns="c", values=["clip", "low"])
    both = [s for s in piv.index if isinstance(piv.loc[s, ("clip", "DMT")], torch.Tensor)
            and isinstance(piv.loc[s, ("clip", "EC")], torch.Tensor)]
    for space in ("clip", "low"):
        m = torch.stack([piv.loc[s, (space, "DMT")] - piv.loc[s, (space, "EC")] for s in both])
        obs, null, p = sign_flip_null(m)
        report["content"][f"DMT-EC_{space}"] = {"n": len(m), "consistency": obs, "p": p}
        group[("DMT-EC", space)] = m.mean(0)
    report["content"]["DMT-EC_clip"]["top"] = cs.top(group[("DMT-EC", "clip")], 10)
    (OUTPUTS / f"dmt_events_report_{kind}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    gen = EmbeddingToImage()
    low = LowLevel(gen.dev)
    i2i = AutoPipelineForImage2Image.from_pipe(gen.pipe)
    i2i.set_progress_bar_config(disable=True)
    concepts = Concepts()

    def mixed(v_clip, low_img, seed=0):
        e = as_condition(v_clip).reshape(1, 1, -1).to(gen.dev, torch.float16)
        return i2i(prompt="", image=low_img, strength=0.75, num_inference_steps=4, guidance_scale=0.0,
                   ip_adapter_image_embeds=[e], generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    def quad(tag, v_clip, v_low):
        li = low.image(v_low)
        return [label(gen(as_condition(v_clip)), f"{tag} content", 16),
                label(gen(as_condition(concepts.retrieval(v_clip))), f"{tag} content (retr.)", 16),
                label(li.resize((512, 512), Image.BICUBIC), f"{tag} color/layout", 16),
                label(mixed(v_clip, li), f"{tag} mixed", 16)]

    out = OUTPUTS / "events"
    out.mkdir(exist_ok=True)
    grows = [quad(c, group[(c, "clip")], group[(c, "low")]) for c in ("DMT", "EC", "EO", "DMT-EC") if (c, "clip") in group]
    sheet(grows).save(out / f"group_{kind}.jpg", quality=90)

    scales = load_scales()
    order = sorted(both, key=lambda s: -(scales.loc[s, "5D-ComplexImagery"] if s in scales.index else -1))
    srows = []
    for s in order:
        row = []
        for c in ("DMT", "EC"):
            vc, vl = piv.loc[s, ("clip", c)], piv.loc[s, ("low", c)]
            li = low.image(vl)
            imgs = [gen(as_condition(vc)), li.resize((512, 512), Image.BICUBIC), mixed(vc, li)]
            for im, t in zip(imgs, ("content", "color/layout", "mixed")):
                im.save(out / f"{s}_{c}_{t.replace('/', '-')}_{kind}.png")
                row.append(label(im, f"{s} {c} {t}", 16))
        srows.append(row)
        print(s, "done", flush=True)
    sheet(srows).save(out / f"subjects_{kind}.jpg", quality=88)


if __name__ == "__main__":
    main()
