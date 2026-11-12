"""Can a THINGS-trained decoder tell which of 10 images a person in another lab
is seeing (perception) or imagining (imagery)?

1. Zero-shot identification: per subject and phase, the mean decoded
   embedding per image (centered across the 10 images) is matched against
   the CLIP embeddings of the 10 actual stimulus pictures (also centered).
   Reports 10-way and within-category accuracy and the rank of the correct
   picture, against chance.
2. Within-lab ceiling: per subject, a cross-validated logistic regression on
   the same trials (band power, or the decoded embedding as features) tells
   whether the information is present in this EEG at all.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from common import DATA, OUTPUTS, SFREQ, WINDOW
from model_variants import build, run

STIM = {"bird": "Animal_bird", "dog": "Animal_dog", "fish": "Animal_fish", "circle": "Figure_circle",
        "pentagram": "Figure_pentagram", "square": "Figure_square", "chair": "Object_chair",
        "cup": "Object_cup", "scissor": "Object_scissor", "watch": "Object_watch"}
TASK_OF = {"bird": "AVI", "dog": "AVI", "fish": "AVI", "circle": "FVI", "pentagram": "FVI", "square": "FVI",
           "chair": "OVI", "cup": "OVI", "scissor": "OVI", "watch": "OVI"}
IMAGES = list(STIM)
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


@torch.no_grad()
def stimulus_embeddings():
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder")
    proc = CLIPImageProcessor()
    ims = [Image.open(DATA / f"raw/imagery/{STIM[k]}.jpg").convert("RGB") for k in IMAGES]
    e = enc(**proc(ims, return_tensors="pt")).image_embeds.float()
    return F.normalize(e - e.mean(0), dim=-1)


def identification(means, stim):
    """means: (10, d) centered embeddings in IMAGES order."""
    sim = F.normalize(means, dim=-1) @ stim.T
    ranks = (sim > sim.diag()[:, None]).sum(1) + 1
    within = []
    for i, k in enumerate(IMAGES):
        same = [j for j, k2 in enumerate(IMAGES) if TASK_OF[k2] == TASK_OF[k]]
        within.append(int(sim[i, same].argmax() == same.index(i)))
    return {"acc10": float((ranks == 1).float().mean()), "mean_rank": float(ranks.float().mean()),
            "acc_within": float(np.mean(within))}


def band_power(x):
    f = np.fft.rfftfreq(WINDOW, 1 / SFREQ)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    return np.log(np.stack([p[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1) + 1e-8).reshape(len(x), -1)


def cv_acc(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    k = min(5, np.bincount(np.unique(y, return_inverse=True)[1]).min())
    return float(cross_val_score(clf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0)).mean())


def main():
    w = np.load(DATA / "processed/imagery_windows.npz")
    x = w["x"]
    subj, ses, task, image, phase, trial = (w[k] for k in ("subject", "session", "task", "image", "phase", "trial"))
    stim = stimulus_embeddings()
    subjects = sorted(set(subj))
    report = {"chance_acc10": 0.1, "chance_mean_rank": 5.5, "chance_within": float(np.mean(
        [1 / sum(TASK_OF[k2] == TASK_OF[k] for k2 in IMAGES) for k in IMAGES]))}

    for variant in ("original", "adabn"):
        m = build("encoder_sub-08_vi.pt", variant, x[phase == "rest"])
        emb = run(m, x)
        for ph in ("perception", "imagery"):
            per_subject, group = [], []
            for s in subjects:
                means = []
                for k in IMAGES:
                    sel = (subj == s) & (phase == ph) & (image == k)
                    means.append(emb[torch.from_numpy(sel)].mean(0) if sel.any() else None)
                if any(v is None for v in means):
                    continue
                means = torch.stack(means)
                means = means - means.mean(0)
                group.append(means)
                per_subject.append(identification(means, stim))
            accs = np.array([r["acc10"] for r in per_subject])
            ranks = np.array([r["mean_rank"] for r in per_subject])
            within = np.array([r["acc_within"] for r in per_subject])
            report[f"{variant}_{ph}"] = {
                "n_subjects": len(per_subject),
                "acc10_mean": float(accs.mean()), "within_mean": float(within.mean()),
                "mean_rank_mean": float(ranks.mean()),
                "p_rank_below_chance": float(wilcoxon(ranks - 5.5, alternative="less").pvalue),
                "group_average": identification(torch.stack(group).mean(0), stim),
            }
            print(variant, ph, json.dumps(report[f"{variant}_{ph}"]), flush=True)
        if variant == "original":
            emb_orig = emb

    ceiling = {}
    for ph in ("perception", "imagery"):
        rows = {"bandpower": [], "atm_embedding": []}
        for s in subjects:
            for t in ("AVI", "FVI", "OVI"):
                sel = (subj == s) & (phase == ph) & (task == t)
                if sel.sum() < 30:
                    continue
                keys = np.char.add(np.char.add(ses[sel], "_"), trial[sel].astype(str))
                uk, inv = np.unique(keys, return_inverse=True)
                y = np.array([image[sel][inv == i][0] for i in range(len(uk))])
                bp = band_power(x[sel])
                eb = emb_orig[torch.from_numpy(sel)].numpy()
                Xbp = np.stack([bp[inv == i].mean(0) for i in range(len(uk))])
                Xeb = np.stack([eb[inv == i].mean(0) for i in range(len(uk))])
                if len(set(y)) < 2 or np.bincount(np.unique(y, return_inverse=True)[1]).min() < 3:
                    continue
                chance = 1 / len(set(y))
                rows["bandpower"].append(cv_acc(Xbp, y) - chance)
                rows["atm_embedding"].append(cv_acc(Xeb, y) - chance)
        ceiling[ph] = {k: {"acc_minus_chance_mean": float(np.mean(v)), "n": len(v),
                           "p_above_chance": float(wilcoxon(v, alternative="greater").pvalue)}
                       for k, v in rows.items()}
        print("ceiling", ph, json.dumps(ceiling[ph]), flush=True)
    report["within_lab_ceiling"] = ceiling
    (OUTPUTS / "imagery_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
