from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
for _d in (DATA / "raw", DATA / "processed", MODELS, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

THINGS_CHANNELS = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
]

SHARED_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F8', 'FC1', 'FC2', 'Cz', 'C3', 'C4', 'T7', 'T8',
    'CPz', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'O1', 'O2',
]

SFREQ = 250
WINDOW = 250


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def normalize_windows(x):
    """Per-window common average reference, demean and per-channel z-score.

    Applied identically to THINGS epochs and DMT windows so both domains
    reach the encoder with the same scale."""
    x = x - x.mean(axis=-2, keepdims=True)
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (x.std(axis=-1, keepdims=True) + 1e-6)


class ChannelTransformerEncoder(nn.Module):
    """Re-implementation of ATM-S (Li et al., NeurIPS 2024) for an arbitrary
    channel count: channel-wise transformer, ShallowNet-style temporal-spatial
    conv, projection to the CLIP ViT-H/14 image-embedding space (1024-d)."""

    def __init__(self, n_channels, seq_len=WINDOW, d_model=250, emb_dim=1024):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.channel_embedding = nn.Parameter(torch.zeros(1, n_channels, d_model))
        self.dropout = nn.Dropout(0.25)
        self.transformer = nn.TransformerEncoderLayer(
            d_model, nhead=5, dim_feedforward=256, dropout=0.25,
            activation="gelu", batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (n_channels, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv2d(40, 40, 1),
        )
        flat = 40 * ((d_model - 24 - 51) // 5 + 1)
        self.proj_in = nn.Linear(flat, emb_dim)
        self.proj_res = nn.Sequential(nn.GELU(), nn.Linear(emb_dim, emb_dim), nn.Dropout(0.5))
        self.proj_norm = nn.LayerNorm(emb_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, x):
        h = self.value_embedding(x) + self.channel_embedding
        h = self.norm(self.transformer(self.dropout(h)))
        h = self.tsconv(h.unsqueeze(1)).flatten(1)
        h = self.proj_in(h)
        return self.proj_norm(h + self.proj_res(h))


def clip_loss(a, b, logit_scale):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = logit_scale.exp() * a @ b.T
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
