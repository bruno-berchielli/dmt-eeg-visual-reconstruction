"""Category readout for a decoded vector: which kinds of image it leans to.

Seven categories described by text probes (OpenCLIP ViT-H/14 text tower, the
space the decoder writes into). A vector's score for a category is its mean
cosine to that category's probes, centered across categories; the weights
shown are a softmax of those scores with a temperature fixed from the spread
of scores over all DMT recordings, so they are relative, not probabilities.
Probe embeddings are cached in models/category_probes.pt.
"""
import torch
import torch.nn.functional as F
from PIL import ImageDraw, ImageFont

from common import DATA, MODELS

NAMES = ("geometry", "architecture", "entities", "people", "animals", "nature", "objects")
ANIMALS = ["a dog", "a cat", "a bear", "a deer", "a bird", "a fish", "a lizard", "an insect", "a horse",
           "a furry mammal in the wild"]
COLORS = {"geometry": (255, 190, 60), "architecture": (255, 140, 90), "entities": (200, 130, 255),
          "people": (255, 120, 170), "animals": (120, 200, 120), "nature": (90, 170, 110), "objects": (150, 170, 200)}


def probes():
    path = MODELS / "category_probes.pt"
    if path.exists():
        return torch.load(path)
    import open_clip
    from stage_kinetics import EXTRA
    from test_text_probes import PROBES
    cats = {**PROBES, **EXTRA, "animals": ANIMALS}
    name = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model, _, _ = open_clip.create_model_and_transforms(name)
    tok = open_clip.get_tokenizer(name)
    with torch.no_grad():
        out = {k: F.normalize(model.eval().encode_text(tok(cats[k])).float(), dim=-1) for k in NAMES}
    torch.save(out, path)
    return out


class Categories:
    def __init__(self):
        self.pr = probes()
        self.names = list(self.pr)
        d = torch.load(DATA / "processed/dmt_dense.pt", weights_only=False)
        pool = torch.cat([r["v"].float().reshape(-1, 50, 1024).mean(1) if len(r["v"]) % 50 == 0 else
                          r["v"].float()[:len(r["v"]) // 50 * 50].reshape(-1, 50, 1024).mean(1)
                          for k, r in d.items() if k.endswith("_DMT")])
        self.temp = float(self.scores(pool).std())

    def scores(self, v):
        s = torch.stack([(F.normalize(v, dim=-1) @ self.pr[k].T).mean(-1) for k in self.names], -1)
        return s - s.mean(-1, keepdim=True)

    def weights(self, v):
        return torch.softmax(self.scores(v) / self.temp, -1)

    def top(self, v, k=3):
        w = self.weights(v)
        return [(self.names[i], float(w[i])) for i in w.topk(k).indices]


def draw(img, top, dim=1.0):
    """Bars for the top categories on a strip at the bottom of the frame."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=17)
    h = 24 * len(top) + 10
    y0 = img.height - h
    d.rectangle([0, y0, img.width, img.height], fill=(0, 0, 0))
    for i, (name, w) in enumerate(top):
        y = y0 + 6 + 24 * i
        col = tuple(int(c * dim + 90 * (1 - dim)) for c in COLORS[name])
        d.text((8, y), f"{name}", fill=col, font=font)
        d.rectangle([130, y + 4, 130 + int(300 * w), y + 16], fill=col)
        d.text((440, y), f"{w:.0%}", fill=col, font=font)
    return img
