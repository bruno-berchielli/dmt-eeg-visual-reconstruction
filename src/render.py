"""Render images, videos and text prompts from the decoded DMT embeddings.

Raw per-subject means collapse onto each decoder's default output, so besides
showing them we render within-subject contrasts (DMT - eyes closed) and the
group-level directions computed by analyze_contrasts.py.

Each contrast is rendered two ways:
  direct    the normalized contrast vector fed to IP-Adapter as is
  retrieval softmax-weighted mix of the nearest real THINGS concept embeddings,
            which keeps the conditioning on the natural-image manifold
"""
import argparse
import json

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from common import DATA, OUTPUTS
from embed_dmt import MODEL_NAMES, concept_names
from sdxl import EmbeddingToImage

CLIP_NORM = 21.5
SCALES = ["5D-ElementaryImagery", "5D-ComplexImagery", "5D-AudioVisualSyn"]


def load_scales():
    df = pd.read_csv(DATA / "raw/dmt/scales_results.csv", index_col=0, decimal=",")
    df.index = [f"S{int(''.join(filter(str.isdigit, str(s)))):02d}" for s in df.index]
    return df[SCALES].apply(pd.to_numeric, errors="coerce")


class Concepts:
    def __init__(self):
        feats = torch.load(DATA / "raw/things/ViT-H-14_features_train.pt")["img_features"].float()
        self.names = concept_names()
        raw = feats.reshape(len(self.names), 10, -1).mean(1)
        self.raw = F.normalize(raw, dim=-1)
        self.centered = F.normalize(raw - raw.mean(0), dim=-1)

    def top(self, v, k=8):
        return [self.names[i] for i in (self.centered @ F.normalize(v, dim=0)).topk(k).indices]

    def retrieval(self, v, k=8, temp=0.02):
        sim = self.centered @ F.normalize(v, dim=0)
        top = sim.topk(k)
        w = torch.softmax(top.values / temp, 0)
        return (w[:, None] * self.raw[top.indices]).sum(0)


def as_condition(v):
    return F.normalize(v, dim=-1) * CLIP_NORM


def label(img, text, size=18):
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, size + 12], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=ImageFont.load_default(size=size))
    return img


def sheet(rows, size=256):
    out = Image.new("RGB", (max(len(r) for r in rows) * size, len(rows) * size), (20, 20, 20))
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            out.paste(im.resize((size, size)), (j * size, i * size))
    return out


def prompt_from(words):
    return "a photo of " + ", ".join(w.rstrip("0123456789").replace("_", " ") for w in words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-subjects", type=int, default=3)
    p.add_argument("--video-window", type=float, default=20.0)
    p.add_argument("--video-step", type=float, default=5.0)
    args = p.parse_args()

    d = torch.load(DATA / "processed/dmt_embeddings.pt", weights_only=False)
    subj, cond, t = d["subject"], d["condition"], d["t_start"]
    scales = load_scales()
    concepts = Concepts()
    gen = EmbeddingToImage()
    img_dir = OUTPUTS / "subjects"
    img_dir.mkdir(parents=True, exist_ok=True)
    prompts = []

    def mask(s, c):
        return torch.from_numpy((subj == s) & (cond == c))

    def n(s, c):
        return int(mask(s, c).sum())

    subjects = [s for s in sorted(set(subj)) if n(s, "DMT") >= 30 and n(s, "EC") >= 30]
    subjects.sort(key=lambda s: -(scales.loc[s, "5D-ComplexImagery"] if s in scales.index else -1))

    group = torch.load(DATA / "processed/group_contrasts.pt")
    residual = torch.load(DATA / "processed/group_residuals.pt")
    grows = []
    for m in MODEL_NAMES:
        e = d["emb"][m]
        eo = torch.stack([e[mask(s, "EO")].mean(0) - e[mask(s, "EC")].mean(0)
                          for s in subjects if n(s, "EO") >= 30]).mean(0)
        row = [label(gen(as_condition(d["noise_emb"][m].mean(0))), f"{m}: noise"),
               label(gen(as_condition(e.mean(0))), f"{m}: grand mean")]
        for name, v in (("DMT-EC", group[m]), ("DMT sem EO", residual[m]), ("EO-EC", eo)):
            row.append(label(gen(as_condition(v)), f"{m}: {name}"))
            row.append(label(gen(as_condition(concepts.retrieval(v))), f"{m}: {name} (retr.)"))
            words = concepts.top(v)
            prompts.append({"scope": "group", "model": m, "contrast": name,
                            "words": words, "prompt": prompt_from(words)})
        grows.append(row)
    sheet(grows).save(OUTPUTS / "group_sheet.jpg", quality=90)

    rows, stats = [], []
    for s in subjects:
        sc = scales.loc[s] if s in scales.index else pd.Series({k: np.nan for k in SCALES})
        row = []
        for m in MODEL_NAMES:
            e = d["emb"][m]
            m_dmt, m_ec = e[mask(s, "DMT")].mean(0), e[mask(s, "EC")].mean(0)
            c = m_dmt - m_ec
            imgs = {"DMT": gen(as_condition(m_dmt)),
                    "DMT-EC": gen(as_condition(c)),
                    "DMT-EC-retr": gen(as_condition(concepts.retrieval(c)))}
            for k, im in imgs.items():
                im.save(img_dir / f"{s}_{m}_{k}.png")
                row.append(label(im, f"{s} {m} {k}"))
            words = concepts.top(c)
            prompts.append({"scope": s, "model": m, "contrast": "DMT-EC",
                            "complex_imagery": sc.get("5D-ComplexImagery"),
                            "words": words, "prompt": prompt_from(words)})
        rows.append(row)
        print(s, "done", flush=True)
    sheet(rows).save(OUTPUTS / "contact_sheet.jpg", quality=88)
    (OUTPUTS / "prompts.json").write_text(json.dumps(prompts, indent=1, ensure_ascii=False, default=float))

    vid_dir = OUTPUTS / "videos"
    vid_dir.mkdir(exist_ok=True)
    for s in subjects[:args.video_subjects]:
        sel = (subj == s) & (cond == "DMT")
        ts = t[sel]
        for m in MODEL_NAMES:
            e = d["emb"][m]
            base = e[mask(s, "EC")].mean(0)
            es = e[torch.from_numpy(sel)]
            frames = []
            for c0 in np.arange(ts.min(), ts.max() - args.video_window + 1e-6, args.video_step):
                w = torch.from_numpy((ts >= c0) & (ts < c0 + args.video_window))
                if w.sum() < 5:
                    continue
                v = concepts.retrieval(es[w].mean(0) - base)
                frames.append(np.asarray(label(gen(as_condition(v), seed=0),
                                               f"{s} {m} DMT-EC t={int(c0)}s", size=22)))
            if frames:
                imageio.mimsave(vid_dir / f"{s}_{m}_DMT.mp4", frames, fps=3, macro_block_size=1)
                print(s, m, "video frames", len(frames), flush=True)


if __name__ == "__main__":
    main()
