"""Run both trained decoders (ATM-S style and linear) over every DMT / EC / EO
window and save the predicted CLIP ViT-H/14 embeddings plus nearest THINGS
concepts per subject and condition."""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import DATA, MODELS, OUTPUTS, SHARED_CHANNELS, ChannelTransformerEncoder, device
from train_linear import LinearDecoder

MODEL_NAMES = ("atm", "linear")


def concept_names():
    root = DATA / "raw/things/images/images_set/training_images"
    return [d.name.split("_", 1)[1] for d in sorted(root.iterdir()) if d.is_dir()]


def load_decoders(sub):
    dev = device()
    ckpt = torch.load(MODELS / f"encoder_sub-{sub}.pt", map_location="cpu")
    atm = ChannelTransformerEncoder(len(SHARED_CHANNELS))
    atm.load_state_dict(ckpt["state_dict"])
    atm.to(dev).eval()
    linear = LinearDecoder.load(MODELS / f"linear_sub-{sub}.pt")

    @torch.no_grad()
    def run_atm(x):
        return torch.cat([atm(torch.from_numpy(x[i:i + 2048]).to(dev)).cpu()
                          for i in range(0, len(x), 2048)])

    def run_linear(x):
        return torch.from_numpy(np.concatenate([linear(x[i:i + 4096]) for i in range(0, len(x), 4096)]))

    return {"atm": run_atm, "linear": run_linear}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub", default="08")
    args = p.parse_args()
    decoders = load_decoders(args.sub)

    w = np.load(DATA / "processed/dmt_windows.npz")
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((2000, len(SHARED_CHANNELS), 250)).astype(np.float32)

    emb = {k: f(w["x"]) for k, f in decoders.items()}
    noise_emb = {k: f(noise) for k, f in decoders.items()}
    torch.save({"emb": emb, "noise_emb": noise_emb, "subject": w["subject"],
                "condition": w["condition"], "t_start": w["t_start"]},
               DATA / "processed/dmt_embeddings.pt")

    feats = torch.load(DATA / "raw/things/ViT-H-14_features_train.pt")["img_features"].float()
    concepts = concept_names()
    concept_feats = F.normalize(feats.reshape(len(concepts), 10, -1).mean(1), dim=-1)

    def top(v, k=8):
        return ", ".join(concepts[i] for i in (concept_feats @ F.normalize(v, dim=-1)).topk(k).indices)

    rows = []
    df = pd.DataFrame({"subject": w["subject"], "condition": w["condition"]})
    for (s, c), idx in df.groupby(["subject", "condition"]).groups.items():
        idx = np.asarray(idx)
        rows.append({"subject": s, "condition": c, "n_windows": len(idx),
                     **{f"{m}_top": top(emb[m][idx].mean(0)) for m in MODEL_NAMES}})
    rows.append({"subject": "noise", "condition": "gaussian", "n_windows": len(noise),
                 **{f"{m}_top": top(noise_emb[m].mean(0)) for m in MODEL_NAMES}})
    pd.DataFrame(rows).to_csv(OUTPUTS / "nearest_concepts.csv", index=False)
    print(pd.DataFrame(rows).to_string(max_colwidth=70))


if __name__ == "__main__":
    main()
