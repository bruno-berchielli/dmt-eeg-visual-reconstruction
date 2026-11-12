"""IV DMT vs placebo (OSF wujgp), eyes closed, 5 midline electrodes.

Per subject: c = (DMT post - DMT pre) - (PLA post - PLA pre), for each
post-injection segment. Tests whether c shares a direction across subjects,
which THINGS concepts it points to, and renders the group directions.
"""
import json

import numpy as np
import torch

from common import DATA, OUTPUTS
from decode import ConceptSpace, embed, group_mean_null, sign_flip_null


def main():
    w = np.load(DATA / "processed/osf_windows.npz")
    emb = embed(w["x"], "encoder_sub-08_osf.pt")
    drug, subj, seg = w["drug"], w["subject"], w["segment"]
    torch.save({"emb": emb, "drug": drug, "subject": subj, "segment": seg, "t": w["t"]},
               DATA / "processed/osf_embeddings.pt")
    cs = ConceptSpace()

    def m(s, dr, sg):
        sel = (subj == s) & (drug == dr) & (seg == sg)
        return emb[torch.from_numpy(sel)].mean(0) if sel.sum() else None

    report, vectors = {}, {}
    for post in ("post1", "post2", "post3"):
        cs_list, dmt_only = [], []
        for s in sorted(set(subj)):
            parts = [m(s, "DMT", post), m(s, "DMT", "pre"), m(s, "PLA", post), m(s, "PLA", "pre")]
            if any(p is None for p in parts):
                continue
            cs_list.append((parts[0] - parts[1]) - (parts[2] - parts[3]))
            dmt_only.append(parts[0] - parts[1])
        c = torch.stack(cs_list)
        obs, null_mean, p = sign_flip_null(c)
        norm, p_norm = group_mean_null(c)
        g = c.mean(0)
        vectors[post] = g
        report[post] = {
            "n_subjects": len(c), "consistency": obs, "null_mean": null_mean, "p_consistency": p,
            "p_group_mean": p_norm, "more_with_dmt": cs.top(g), "less_with_dmt": cs.top(-g),
            "rank_face": cs.rank(g, "face"), "rank_kaleidoscope": cs.rank(g, "kaleidoscope"),
            "rank_stained_glass": cs.rank(g, "stained_glass"), "n_concepts": len(cs.names),
        }
        vectors[post + "_dmt_only"] = torch.stack(dmt_only).mean(0)
    torch.save(vectors, DATA / "processed/osf_group_vectors.pt")
    (OUTPUTS / "osf_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
