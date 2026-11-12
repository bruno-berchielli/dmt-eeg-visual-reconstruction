"""Shared helpers to run a trained encoder over a windows file and to rank
THINGS concepts for a contrast vector."""
import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, MODELS, ChannelTransformerEncoder, device
from embed_dmt import concept_names


@torch.no_grad()
def embed(x, ckpt_name):
    ckpt = torch.load(MODELS / ckpt_name, map_location="cpu")
    model = ChannelTransformerEncoder(len(ckpt["channels"]))
    model.load_state_dict(ckpt["state_dict"])
    dev = device()
    model.to(dev).eval()
    return torch.cat([model(torch.from_numpy(x[i:i + 2048]).to(dev)).cpu()
                      for i in range(0, len(x), 2048)])


class ConceptSpace:
    def __init__(self):
        feats = torch.load(DATA / "raw/things/ViT-H-14_features_train.pt")["img_features"].float()
        self.names = concept_names()
        raw = feats.reshape(len(self.names), 10, -1).mean(1)
        self.centered = F.normalize(raw - raw.mean(0), dim=-1)

    def scores(self, v):
        return self.centered @ F.normalize(v, dim=0)

    def top(self, v, k=12):
        return [self.names[i] for i in self.scores(v).topk(k).indices]

    def rank(self, v, concept):
        s = self.scores(v)
        return int((s > s[self.names.index(concept)]).sum()) + 1


def consistency(c):
    n = F.normalize(c, dim=-1)
    sim = n @ n.T
    return sim[~torch.eye(len(c), dtype=bool)].mean().item()


def sign_flip_null(c, n_perm=1000, seed=0):
    """Null for 'subject contrasts share a direction': random per-subject sign flips."""
    g = torch.Generator().manual_seed(seed)
    obs = consistency(c)
    null = np.array([consistency(c * (torch.randint(0, 2, (len(c), 1), generator=g) * 2 - 1))
                     for _ in range(n_perm)])
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (n_perm + 1))


def group_mean_null(c, n_perm=1000, seed=0):
    """Null for the norm of the mean contrast (one-sample test in embedding space)."""
    g = torch.Generator().manual_seed(seed)
    obs = c.mean(0).norm().item()
    null = np.array([(c * (torch.randint(0, 2, (len(c), 1), generator=g) * 2 - 1)).mean(0).norm().item()
                     for _ in range(n_perm)])
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1))


def load_psiconnect():
    """Windows plus string metadata (older windows files stored metadata as
    object arrays; psiconnect_meta.npz holds a plain-dtype copy)."""
    w = np.load(DATA / "processed/psiconnect_windows.npz")
    meta_path = DATA / "processed/psiconnect_meta.npz"
    meta = np.load(meta_path) if meta_path.exists() else w
    return w["x"], {k: meta[k] for k in ("subject", "session", "task", "t")}
