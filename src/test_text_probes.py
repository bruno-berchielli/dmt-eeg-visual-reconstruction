"""Do the decoded DMT contrasts point toward psychedelic imagery described in
free text ("a mandala", "a fractal", "an alien entity") more than toward
neutral descriptions, and more than the eyes-open contrast or the
spectral-transplant contrast (same spectral change, no DMT)?

Uses the OpenCLIP ViT-H/14 text tower, whose joint space is the one the
decoders write into (same model as the IP-Adapter image encoder).
"""
import json

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from common import DATA, OUTPUTS
from embed_dmt import MODEL_NAMES, load_decoders
from test_spectral_transplant import transplant

PROBES = {
    "geometry": ["a mandala", "a fractal pattern", "sacred geometry", "a kaleidoscopic pattern",
                 "an intricate symmetric geometric pattern", "a psychedelic geometric pattern",
                 "a tunnel of light", "a spiraling vortex", "an ornate tessellation", "a glowing grid"],
    "entities": ["an alien entity", "a machine elf", "a divine being", "a strange otherworldly creature",
                 "an angel made of light", "a multi-armed deity", "a humanoid alien", "a jester",
                 "a spirit", "a serpent deity"],
    "people": ["a man", "a woman", "a person standing", "a child", "a group of people",
               "a face", "a man raising his arm", "a person in a costume", "a dancer", "a crowd"],
    "objects": ["a chair", "a cup", "a car", "a lamp", "a book", "a shoe", "a spoon",
                "a bicycle", "a bottle", "a table"],
    "nature": ["moss", "gravel", "tree bark", "a forest", "grass", "a rock", "leaves",
               "a field", "mushrooms", "a river"],
}


def text_embeddings():
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    tok = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model.eval()
    out = {}
    with torch.no_grad():
        for k, texts in PROBES.items():
            out[k] = F.normalize(model.encode_text(tok(texts)).float(), dim=-1)
    allv = torch.cat(list(out.values()))
    center = allv.mean(0)
    return {k: F.normalize(v - center, dim=-1) for k, v in out.items()}


def main():
    probes = text_embeddings()
    w = np.load(DATA / "processed/dmt_windows.npz")
    x, subj, cond = w["x"], w["subject"], w["condition"]
    decoders = load_decoders("08")

    def n(s, c):
        return int(((subj == s) & (cond == c)).sum())

    subjects = [s for s in sorted(set(subj)) if min(n(s, "DMT"), n(s, "EC"), n(s, "EO")) >= 30]
    report = {"n_subjects": len(subjects)}
    for m in MODEL_NAMES:
        dec = decoders[m]
        contrasts = {"DMT-EC": [], "EO-EC": [], "transplant-EC": []}
        for s in subjects:
            ec = x[(subj == s) & (cond == "EC")]
            dmt = x[(subj == s) & (cond == "DMT")]
            e_ec = dec(ec).mean(0)
            contrasts["DMT-EC"].append(dec(dmt).mean(0) - e_ec)
            contrasts["EO-EC"].append(dec(x[(subj == s) & (cond == "EO")]).mean(0) - e_ec)
            contrasts["transplant-EC"].append(dec(transplant(ec, dmt)).mean(0) - e_ec)
        scores = {}
        for name, cs in contrasts.items():
            c = F.normalize(torch.stack(cs), dim=-1)
            scores[name] = {k: (c @ v.T).mean(1).numpy() for k, v in probes.items()}
        res = {}
        for name, sc in scores.items():
            res[name] = {k: float(np.mean(v)) for k, v in sc.items()}
        for k in ("geometry", "entities", "people", "nature"):
            rel = {name: scores[name][k] - scores[name]["objects"] for name in scores}
            res[f"{k}_minus_objects"] = {
                "DMT-EC_mean": float(rel["DMT-EC"].mean()),
                "p_DMT-EC_vs_0": float(wilcoxon(rel["DMT-EC"]).pvalue),
                "p_DMT-EC_vs_EO-EC": float(wilcoxon(rel["DMT-EC"], rel["EO-EC"]).pvalue),
                "EO-EC_mean": float(rel["EO-EC"].mean()),
                "p_DMT-EC_vs_transplant": float(wilcoxon(rel["DMT-EC"], rel["transplant-EC"]).pvalue),
                "transplant_mean": float(rel["transplant-EC"].mean()),
            }
        report[m] = res
    (OUTPUTS / "text_probe_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
