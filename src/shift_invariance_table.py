"""Evaluate the original ATM-S encoder under random per-trial time shifts and
merge it with shift_invariance_report.json and the shift_robust_{cnn,atm}
reports into one table (100-way on odd test images; chance 1% / 5%).

Writes outputs/shift_invariance_table.json.
Usage: python src/shift_invariance_table.py
"""
import json

import numpy as np
import torch

from common import DATA, MODELS, OUTPUTS, SHARED_CHANNELS, ChannelTransformerEncoder, device
from train_encoder import load_features
from train_shift_robust import evaluate, test_sets


def main():
    dev = device()
    feats = load_features("test")
    tests = test_sets()
    model = ChannelTransformerEncoder(len(SHARED_CHANNELS))
    model.load_state_dict(torch.load(MODELS / "encoder_sub-08.pt", map_location="cpu")["state_dict"])
    x_tr = np.load(DATA / "processed" / "things08_training_21ch.npy", mmap_mode="r")
    pick = np.sort(np.random.default_rng(1).choice(len(x_tr), 4096, replace=False))
    center = torch.from_numpy(np.ascontiguousarray(x_tr[pick, 0]))
    atm = evaluate(model.to(dev), tests, feats, dev, center)

    rows = {"ATM-S original (aligned training)": {
        "aligned (waveform avg)": atm["aligned_waveform_avg"]["eval"],
        "aligned (per-trial emb avg)": atm["aligned"]["eval"],
        "aligned single": atm["aligned"]["eval_single"],
        "shifted (per-trial emb avg)": atm["shifted"]["eval"],
        "shifted single": atm["shifted"]["eval_single"]}}

    ridge = json.loads((OUTPUTS / "shift_invariance_report.json").read_text())
    for name, r in ridge["results"].items():
        row = {}
        for cond, v in r.items():
            tag = "aligned (waveform avg)" if "waveform" in name else f"{cond} (per-trial feat avg)"
            row[tag] = v["eval_avg"]
            row[f"{cond} single"] = v["eval_single"]
            row[f"{cond} lambda"] = v["lambda"]
        rows[f"ridge: {name}"] = row

    for arch in ("cnn", "atm"):
        p = OUTPUTS / f"shift_robust_{arch}_report.json"
        if p.exists():
            b = json.loads(p.read_text())["best"]
            rows[f"learned shift-robust {arch} (epoch {b['epoch']})"] = {
                "aligned (waveform avg)": b["aligned_waveform_avg"]["eval"],
                "aligned (per-trial emb avg)": b["aligned"]["eval"],
                "aligned single": b["aligned"]["eval_single"],
                "shifted (per-trial emb avg)": b["shifted"]["eval"],
                "shifted single": b["shifted"]["eval_single"]}

    out = {"protocol": "100-way retrieval on the 100 odd THINGS test images (chance top1 0.01, top5 0.05); "
                       "hyperparameters/epochs chosen on the 100 even images. Shifted = independent uniform "
                       "circular shift (0-249 samples) per trial before featurizing/embedding; "
                       "'single' = mean over the 80 single-trial retrievals. All embeddings are centered "
                       "(ridge intercept removed; network outputs minus their mean on training trials).",
           "best_invariant_ridge": ridge.get("best_invariant_by_selection_half"),
           "rows": rows}
    (OUTPUTS / "shift_invariance_table.json").write_text(json.dumps(out, indent=1))
    for k, v in rows.items():
        print(k, {c: (round(m["top1"], 3), round(m["top5"], 3)) for c, m in v.items() if isinstance(m, dict)})


if __name__ == "__main__":
    main()
