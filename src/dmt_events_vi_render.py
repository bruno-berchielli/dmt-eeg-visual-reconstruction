"""Images for the event-locked DMT decoding with the imagery-lab model
(dmt_events_vi.py). The decoder returns scores for 10 pictures (3 geometric
figures, 3 animals, 4 objects); per recording the event-locked component is
the mean score at events minus the mean at the surrogate's events.

  direct   score-weighted sum of the 10 pictures' centered CLIP embeddings
  mixture  softmax-weighted mix of the pictures' own CLIP embeddings

Also writes the category means per condition.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import wilcoxon
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from analyze_imagery import IMAGES, STIM, TASK_OF
from common import DATA, OUTPUTS
from render import as_condition, label, load_scales, sheet
from sdxl import EmbeddingToImage

CAT = {"FVI": "figures", "AVI": "animals", "OVI": "objects"}


@torch.no_grad()
def raw_stim():
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder")
    ims = [Image.open(DATA / f"raw/imagery/{STIM[k]}.jpg").convert("RGB") for k in IMAGES]
    return enc(**CLIPImageProcessor()(ims, return_tensors="pt")).image_embeds.float()


def main():
    d = torch.load(DATA / "processed/dmt_events_vi.pt", weights_only=False)
    rows = []
    for v in d.values():
        if v["events"] is None or v["surrogate_mean"] is None or len(v["events_s"]) < 10:
            continue
        rows.append({"s": v["subject"], "c": v["condition"], "n": len(v["events_s"]),
                     "e": v["events"].mean(0) - v["surrogate_mean"]})
    df = pd.DataFrame(rows)
    stim = raw_stim()
    stim_c = F.normalize(stim - stim.mean(0), dim=-1)
    report = {}
    group = {c: torch.stack(list(g.e)).mean(0) for c, g in df.groupby("c")}
    piv = df.pivot(index="s", columns="c", values="e")
    both = [s for s in piv.index if isinstance(piv.loc[s, "DMT"], torch.Tensor) and isinstance(piv.loc[s, "EC"], torch.Tensor)]
    group["DMT-EC"] = torch.stack([piv.loc[s, "DMT"] - piv.loc[s, "EC"] for s in both]).mean(0)
    for c, e in group.items():
        report[c] = {"scores": {k: round(float(x), 4) for k, x in zip(IMAGES, e)},
                     "categories": {name: round(float(np.mean([e[i] for i, k in enumerate(IMAGES) if CAT[TASK_OF[k]] == name])), 4)
                                    for name in CAT.values()}}
    for name in CAT.values():
        cols = [i for i, k in enumerate(IMAGES) if CAT[TASK_OF[k]] == name]
        diff = [float((piv.loc[s, "DMT"] - piv.loc[s, "EC"])[cols].mean()) for s in both]
        report.setdefault("paired_DMT_vs_EC", {})[name] = {"n": len(diff), "mean": float(np.mean(diff)),
                                                           "p": float(wilcoxon(diff).pvalue)}
    (OUTPUTS / "dmt_events_vi_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    gen = EmbeddingToImage()

    def pair(tag, e):
        direct = (e[:, None] * stim_c).sum(0)
        mix = (torch.softmax(e / (e.std() + 1e-9) * 1.5, 0)[:, None] * stim).sum(0)
        top = IMAGES[int(e.argmax())]
        return [label(gen(as_condition(direct)), f"{tag} direct", 16), label(gen(as_condition(mix)), f"{tag} mixture ({top})", 16)]

    out = OUTPUTS / "events"
    out.mkdir(exist_ok=True)
    grid = [sum((pair(c, group[c]) for c in ("DMT", "EC")), []), sum((pair(c, group[c]) for c in ("EO", "DMT-EC")), [])]
    sheet(grid).save(out / "group_vi.jpg", quality=90)
    scales = load_scales()
    order = sorted(both, key=lambda s: -(scales.loc[s, "5D-ComplexImagery"] if s in scales.index else -1))
    sheet([pair(f"{s} DMT", piv.loc[s, "DMT"]) + pair(f"{s} EC", piv.loc[s, "EC"]) for s in order]).save(
        out / "subjects_vi.jpg", quality=88)


if __name__ == "__main__":
    main()
