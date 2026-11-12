"""Is the decoded DMT "content" just a power-spectrum change?

For each subject, take the eyes-closed (EC) windows and rescale every
frequency bin of every channel so the average power spectrum matches that
subject's DMT windows, keeping the EC phases (i.e. the waveform details).
If decoding these spectrally-transplanted EC windows reproduces the real
DMT-EC direction, the decoders are reading spectral shape, not content.

Also renders contrasts from label-permuted windows to check whether "people"
appear even for meaningless directions (generator/encoder bias).
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, OUTPUTS, normalize_windows
from decode import ConceptSpace, consistency
from embed_dmt import load_decoders


def mean_power(x):
    return (np.abs(np.fft.rfft(x, axis=-1)) ** 2).mean(0)


def transplant(ec, dmt):
    gain = np.sqrt(mean_power(dmt) / (mean_power(ec) + 1e-12))
    spec = np.fft.rfft(ec, axis=-1) * gain[None]
    return normalize_windows(np.fft.irfft(spec, n=ec.shape[-1], axis=-1).astype(np.float32))


def main():
    w = np.load(DATA / "processed/dmt_windows.npz")
    x, subj, cond = w["x"], w["subject"], w["condition"]
    decoders = load_decoders("08")
    cs = ConceptSpace()
    rng = np.random.default_rng(0)

    def n(s, c):
        return int(((subj == s) & (cond == c)).sum())

    subjects = [s for s in sorted(set(subj)) if n(s, "DMT") >= 30 and n(s, "EC") >= 30]
    report = {}
    perm_vectors = {}
    for m, dec in decoders.items():
        real, fake, perm = [], [], []
        for s in subjects:
            ec, dmt = x[(subj == s) & (cond == "EC")], x[(subj == s) & (cond == "DMT")]
            e_ec, e_dmt = dec(ec), dec(dmt)
            e_fake = dec(transplant(ec, dmt))
            real.append(e_dmt.mean(0) - e_ec.mean(0))
            fake.append(e_fake.mean(0) - e_ec.mean(0))
            both = torch.cat([e_dmt, e_ec])[torch.from_numpy(rng.permutation(len(e_dmt) + len(e_ec)))]
            perm.append(both[:len(e_dmt)].mean(0) - both[len(e_dmt):].mean(0))
        real, fake, perm = map(torch.stack, (real, fake, perm))
        per_subject = F.cosine_similarity(real, fake, dim=1)
        resid = real - fake
        report[m] = {
            "cos_group_real_vs_transplant": F.cosine_similarity(real.mean(0), fake.mean(0), dim=0).item(),
            "cos_subject_real_vs_transplant_median": per_subject.median().item(),
            "norm_ratio_transplant_over_real": (fake.norm(dim=1) / real.norm(dim=1)).median().item(),
            "transplant_top": cs.top(fake.mean(0)),
            "real_top": cs.top(real.mean(0)),
            "residual_consistency": consistency(resid),
            "residual_top": cs.top(resid.mean(0)),
            "permuted_consistency": consistency(perm),
            "permuted_top_examples": [cs.top(v, 5) for v in perm[:4]],
        }
        perm_vectors[m] = {"perm": perm[:4], "transplant": fake.mean(0), "residual": resid.mean(0)}
    torch.save(perm_vectors, DATA / "processed/bias_test_vectors.pt")
    (OUTPUTS / "spectral_transplant_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
