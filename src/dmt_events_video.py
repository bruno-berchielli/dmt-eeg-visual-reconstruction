"""Video of one recording's event-locked decoding over time (content only).

Each frame is a Gaussian-weighted mean (in time) of the vectors decoded at
the detected events, minus the mean at the surrogate's events, rendered with
IP-Adapter + SDXL-Turbo. Same seed in every frame and no image prior, so
changes come from the vector.

With "halves", the events are split into two independent sets (odd / even in
time) and rendered side by side on the same clock: what both halves show is
in the data, what only one shows is sampling noise. Also prints how well the
two trajectories agree (see halves_report).

With "dense", the vectors come from dmt_dense.py (a window at every local
maximum of the detector, about 8 times more data per frame) and the kernel
is narrowed to 20 s.

With "fade" (implies dense), each frame is shown only as far as the data
agree on it: the windows are split into two halves by random 10 s blocks,
20 times; the cosine between the two halves' local vectors, averaged over
the splits, is the frame's agreement. The picture (from all the data) is
blended with static in proportion: full picture at agreement >= FULL, pure
static at <= 0.

With "prior" (implies dense), low-agreement frames fall back, instead of
static, on what the drug level predicts for this person: a regression of
the person's own window vectors on the measured level (alpha suppression
relative to their eyes closed, drug_curve.py), v = a + b * level. Frame =
agreement-weighted mix of the local data and that prediction. Nothing from
reports enters; the prior is the person's own state-to-vector mapping.

With "calibrated" (implies dense), both sources get a confidence measured
against other people's recordings, and static returns when neither holds
(see calibrated_confidence):
  data    agreement between two halves of the recording at the same moment
  prior   agreement between the state prediction fitted on one half and the
          other half's local data
Frame vector = data-confidence mix of local data and prediction; the image
is visible in proportion to the larger of the two confidences.

With "lens" (implies calibrated), the width of the time window adapts per
frame: the narrowest of LENS (seconds each side) whose data agreement
reaches z >= Z_LENS is used, so strong stretches are shown at fine time
resolution and weak ones borrow from a wider neighbourhood. The chosen
width is median-smoothed over time. Where no width passes, the widest one
is mixed with the state prediction as in the calibrated mode.

With "inferred" (implies calibrated), the window width comes from
lens_model.py: per moment, the smallest window expected to reach the target
reliability (target=0.65 by default) given the local consistency of the
decoded windows. No list of widths and no acceptance threshold.

In the calibrated, lens and inferred modes the bottom strip shows the three categories the
frame's vector leans to and their relative weights (categories.py), greyed
when the frame is mostly static.

Usage: python dmt_events_video.py S28 [T500multi] [DMT] [halves|dense|fade|prior|calibrated|lens|inferred] [step=2] [sigma=20] [target=0.65] [max=30] [tmax=SECONDS] [name=NAME] [contrast]
contrast (dense modes): every window vector has the mean vector of the same person's eyes-closed recording
subtracted, so frames show what differs from that person's rest; the other people used for the
confidence floor get the same subtraction.
max: ceiling of the inferred lens in seconds (default 30; where that is not enough the frame shows its
uncertainty). tmax: use only the first SECONDS of the recording. name: label and file name carry NAME
instead of subject and condition (outputs/videos/NAME.mp4).
step: seconds of recording per frame (playback speed is kept); sigma: window width for the dense modes.
"""
import json
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import DATA, OUTPUTS
from render import as_condition, label

SIGMA = 30.0
STEP = 2.0
FPS = 6
FULL = 0.6


def load(subj, kind="T500multi", cond="DMT", dense=False):
    if dense:
        d = torch.load(DATA / "processed/dmt_dense.pt", weights_only=False)[f"dmt_{subj}_{cond}"]
        return d["t"], d["v"].float()
    v = torch.load(DATA / f"processed/dmt_events_{kind}.pt", weights_only=False)[f"dmt_{subj}_{cond}"]
    t = torch.as_tensor(np.asarray(v["events_s"])).float()
    return t, v["events_clip"].float() - v["surrogate_clip_mean"]


def trajectory(t, x, clock, sigma=SIGMA):
    w = torch.exp(-0.5 * ((t[None] - clock[:, None]) / sigma) ** 2)
    return (w / w.sum(1, keepdim=True)) @ x


def agreement(t, x, clock, sigma, n_splits=20, seed=0):
    w = torch.exp(-0.5 * ((t[None] - clock[:, None]) / sigma) ** 2)
    rng = np.random.default_rng(seed)
    block = (t // 10).long()
    out = []
    for _ in range(n_splits):
        a = torch.from_numpy(rng.permutation(int(block.max()) + 1) % 2 == 0)[block]
        A = (w[:, a] / w[:, a].sum(1, keepdim=True)) @ x[a]
        B = (w[:, ~a] / w[:, ~a].sum(1, keepdim=True)) @ x[~a]
        out.append(F.cosine_similarity(A, B, dim=1))
    return torch.stack(out).mean(0)


def drug_level(subj, cond):
    from drug_curve import BIN
    curves = np.load(DATA / "processed/drug_curve.npz")
    cur, ec = curves[f"{subj}_{cond}"], curves[f"{subj}_EC"]
    tb = (np.arange(len(cur)) + 0.5) * BIN
    return lambda at: torch.from_numpy(-np.interp(at.numpy(), tb, np.log(cur[:, 0] / np.median(ec[:, 0])))).float()


def state_prior(level, t, x, clock):
    X = torch.stack([torch.ones(len(t)), level(t)], 1)
    coef = torch.linalg.lstsq(X, x).solution
    return torch.stack([torch.ones(len(clock)), level(clock)], 1) @ coef


N_COMP = 50
Z_STATIC, Z_FULL = 1.0, 3.0
LENS = (5.0, 10.0, 15.0, 20.0, 25.0, 40.0)
Z_LENS = 2.0


def _halves(t, x, clock, sigma, n_splits, seed):
    w = torch.exp(-0.5 * ((t[None] - clock[:, None]) / sigma) ** 2)
    rng = np.random.default_rng(seed)
    block = (t // 10).long()
    for _ in range(n_splits):
        a = torch.from_numpy(rng.permutation(int(block.max()) + 1) % 2 == 0)[block]
        yield a, (w[:, a] / w[:, a].sum(1, keepdim=True)) @ x[a], (w[:, ~a] / w[:, ~a].sum(1, keepdim=True)) @ x[~a]


CONTRAST = False


def dense_pool(cond):
    """All dense recordings of a condition; with CONTRAST, minus each person's eyes-closed mean."""
    d = torch.load(DATA / "processed/dmt_dense.pt", weights_only=False)
    out = {}
    for k, r in d.items():
        if not k.endswith(f"_{cond}"):
            continue
        v = r["v"].float()
        if CONTRAST:
            ec = d.get(k.rsplit("_", 1)[0] + "_EC")
            if ec is None:
                continue
            v = v - ec["v"].float().mean(0)
        out[k] = {"t": r["t"], "v": v}
    return out


def calibrated_confidence(subj, cond, level, t, x, clock, sigma=20.0, n_splits=20):
    """Per frame, confidence in the local data and in the state prediction.

    Cosines are taken after whitening with the first N_COMP principal
    components of all recordings' window vectors: the raw vectors live in
    about 4 effective dimensions, where unrelated recordings reach |cos| of
    0.45 by chance. Each agreement is expressed as z against the cosines
    between this recording's half and halves of OTHER people's recordings.
    Confidence is 0 at z <= Z_STATIC and 1 at z >= Z_FULL.
    """
    # repaired copies ("S27r") are the same person as the original: neither may serve as "another person"
    person = lambda k: k.split("_")[1].rstrip("r")
    others = {k: (r["t"], r["v"]) for k, r in dense_pool(cond).items() if person(k) != subj.rstrip("r")}
    seen = {}
    for k in sorted(others, key=lambda k: not k.split("_")[1].endswith("r")):
        seen.setdefault(person(k), k)
    others = {k: others[k] for k in seen.values()}
    pool = torch.cat([x] + [v for _, v in others.values()])
    mu = pool.mean(0)
    _, S, Vt = torch.linalg.svd(pool - mu, full_matrices=False)
    P = Vt[:N_COMP].T / S[:N_COMP] * np.sqrt(len(pool))
    white = lambda v: (v - mu) @ P
    foreign = []
    for to, vo in others.values():
        co = torch.arange(float(to.min()), float(to.max()) + STEP, STEP)
        foreign.append(F.normalize(next(_halves(to, white(vo), co, sigma, 1, 1))[2], dim=1))
    foreign = torch.cat(foreign)
    xw = white(x)
    same = {"data": [], "prior": []}
    floor = {}
    for i, (a, A, B) in enumerate(_halves(t, xw, clock, sigma, n_splits, 0)):
        for name, src in (("data", A), ("prior", white(state_prior(level, t[a], x[a], clock)))):
            same[name].append(F.cosine_similarity(src, B, dim=1))
            if i == 0:
                floor[name] = (F.normalize(src, dim=1) @ foreign.T).flatten()
    out = {}
    for name in same:
        z = (torch.stack(same[name]).mean(0) - floor[name].mean()) / floor[name].std()
        out[name] = ((z - Z_STATIC) / (Z_FULL - Z_STATIC)).clamp(0, 1)
        out[name + "_z"] = z
    return out


def halves_report(t, x, clock, n_perm=2000, seed=0):
    """Agreement between the trajectories of two independent halves of the events.

    mean_cos       cosine between the halves' time-averaged vectors (overall look)
    same_time_cos  mean cosine, frame by frame, after removing each half's own
                   time average (does the look change the same way over time?)
    p              share of random splits of event TIMES (labels shuffled across
                   time within each half) with same_time_cos at least as high
    """
    a, b = slice(0, None, 2), slice(1, None, 2)
    ta, tb = trajectory(t[a], x[a], clock), trajectory(t[b], x[b], clock)

    def same_time(u, v):
        return float(F.cosine_similarity(u - u.mean(0), v - v.mean(0), dim=1).mean())

    obs = same_time(ta, tb)
    g = torch.Generator().manual_seed(seed)
    xb = x[b]
    null = [same_time(ta, trajectory(t[b], xb[torch.randperm(len(xb), generator=g)], clock)) for _ in range(n_perm)]
    return {"n_events": [len(t[a]), len(t[b])],
            "mean_cos": float(F.cosine_similarity(x[a].mean(0), x[b].mean(0), dim=0)),
            "same_time_cos": obs, "null_mean": float(np.mean(null)),
            "p": float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1))}


def main():
    global STEP
    opts = dict(a.split("=") for a in sys.argv if "=" in a)
    fps = FPS * STEP / float(opts.get("step", STEP))
    STEP = float(opts.get("step", STEP))
    sigma = float(opts.get("sigma", 20.0))
    subj = sys.argv[1]
    kind = sys.argv[2] if len(sys.argv) > 2 else "T500multi"
    cond = sys.argv[3] if len(sys.argv) > 3 else "DMT"
    lens, inferred = "lens" in sys.argv, "inferred" in sys.argv
    fade, prior, calib = "fade" in sys.argv, "prior" in sys.argv, "calibrated" in sys.argv or lens or inferred
    halves, dense = "halves" in sys.argv, "dense" in sys.argv or fade or prior or calib
    t, x = load(subj, kind, cond, dense)
    if "tmax" in opts:
        keep = t <= float(opts["tmax"])
        t, x = t[keep], x[keep]
    global CONTRAST
    CONTRAST = "contrast" in sys.argv
    if CONTRAST:
        x = x - torch.load(DATA / "processed/dmt_dense.pt", weights_only=False)[f"dmt_{subj}_EC"]["v"].float().mean(0)
    tag = opts.get("name", f"{subj} {cond}" + (" - EC" if CONTRAST else ""))
    clock = torch.arange(float(t.min()), float(t.max()) + STEP, STEP)
    if halves:
        rep = halves_report(t, x, clock)
        print(json.dumps(rep, indent=1), flush=True)
        (OUTPUTS / f"halves_{subj}_{cond}_{kind}.json").write_text(json.dumps(rep, indent=1))
        tracks = [("half A", trajectory(t[0::2], x[0::2], clock)), ("half B", trajectory(t[1::2], x[1::2], clock))]
    else:
        tracks = [("content", trajectory(t, x, clock, sigma if dense else SIGMA))]

    alpha = None
    if fade or prior:
        agree = agreement(t, x, clock, sigma)
        alpha = (agree / FULL).clamp(0, 1)
    visible = None
    if calib:
        level = drug_level(subj, cond)
        pred = state_prior(level, t, x, clock)
        width = None
        if lens:
            confs = [calibrated_confidence(subj, cond, level, t, x, clock, sigma=sg, n_splits=10) for sg in LENS]
            z = torch.stack([c["data_z"] for c in confs])
            ok = z >= Z_LENS
            pick = torch.where(ok.any(0), ok.float().argmax(0), torch.full((len(clock),), len(LENS) - 1))
            pad = torch.cat([pick[:1].repeat(5), pick, pick[-1:].repeat(5)])
            pick = pad.unfold(0, 11, 1).median(1).values
            rows = torch.arange(len(clock))
            data = torch.stack([trajectory(t, x, clock, sg) for sg in LENS])[pick, rows]
            cd = torch.stack([c["data"] for c in confs])[pick, rows]
            cp = torch.stack([c["prior"] for c in confs])[pick, rows]
            conf = {"data": cd, "prior": cp, "data_z": z[pick, rows], "prior_z": torch.stack([c["prior_z"] for c in confs])[pick, rows]}
            width = [LENS[i] for i in pick.tolist()]
            print("lens use:", {int(sg): round(float((pick == i).float().mean()), 2) for i, sg in enumerate(LENS)})
        elif inferred:
            import lens_model
            lens_model.TARGET, lens_model.MAX = float(opts.get("target", 0.65)), float(opts.get("max", 30.0))
            white = lens_model.whitener(dense_pool(cond), cond)
            sig, _ = lens_model.infer_lens(t, white(x), clock)
            data = lens_model.adaptive_trajectory(t, x, clock, sig)
            grid = (10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)
            confs = [calibrated_confidence(subj, cond, level, t, x, clock, sigma=sg, n_splits=10) for sg in grid]
            pick = (sig[:, None] - torch.tensor(grid)[None]).abs().argmin(1)
            rows = torch.arange(len(clock))
            conf = {k: torch.stack([c[k] for c in confs])[pick, rows] for k in ("data", "prior", "data_z", "prior_z")}
            cd, cp = conf["data"], conf["prior"]
            width = sig.tolist()
            print("inferred lens: median", round(float(sig.median()), 1), "p10", round(float(sig.quantile(0.1)), 1),
                  "p90", round(float(sig.quantile(0.9)), 1))
        else:
            conf = calibrated_confidence(subj, cond, level, t, x, clock, sigma=sigma)
            data = tracks[0][1]
            cd, cp = conf["data"], conf["prior"]
        tracks = [("", cd[:, None] * F.normalize(data, dim=1) + (1 - cd[:, None]) * F.normalize(pred, dim=1))]
        visible = torch.maximum(cd, cp)
        for k in ("data", "prior"):
            print(k, "mean z", round(float(conf[k + "_z"].mean()), 2), "mean confidence", round(float(conf[k].mean()), 2))
        print("visible: mean", round(float(visible.mean()), 2), "frames above 0.75:", round(float((visible > 0.75).float().mean()), 2),
              "below 0.25:", round(float((visible < 0.25).float().mean()), 2))
    if prior:
        pred = state_prior(drug_level(subj, cond), t, x, clock)
        data = tracks[0][1]
        print("cosine between local data and state prediction: median",
              round(float(F.cosine_similarity(data, pred, dim=1).median()), 2))
        mix = alpha[:, None] * F.normalize(data, dim=1) + (1 - alpha[:, None]) * F.normalize(pred, dim=1)
        tracks = [("", mix)]
    if fade or prior:
        print("agreement: median", round(float(agree.median()), 2), "share of frames at full strength",
              round(float((alpha >= 1).float().mean()), 2), "fully faded", round(float((alpha <= 0).float().mean()), 2))
    cats = None
    if calib:
        from categories import Categories, draw
        cats = Categories()
    from sdxl import EmbeddingToImage
    gen = EmbeddingToImage()
    noise_rng = np.random.default_rng(0)
    out = OUTPUTS / "videos"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{opts['name']}.mp4" if "name" in opts else out / f"{subj}_{cond}_events_{kind}{''.join('_' + a.replace('=', '') for a in sys.argv if '=' in a)}{'_contrast' if CONTRAST else ''}{'_halves' if halves else '_inferred_lens' if inferred else '_lens' if lens else '_calibrated' if calib else '_agreement' if fade else '_prior' if prior else '_dense' if dense else '_content'}.mp4"
    # frames are streamed to the file, so an interrupted run leaves a playable partial video
    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    n_frames = 0
    for i, now in enumerate(clock.tolist()):
        frame = Image.new("RGB", (512 * len(tracks), 512))
        for j, (name, tr) in enumerate(tracks):
            stamp = f"{tag}  {int(now // 60)}:{int(now % 60):02d}  " if j == 0 else ""
            img = gen(as_condition(tr[i]))
            if prior:
                name = f"data {float(alpha[i]):.0%}"
            if fade or calib:
                static = np.clip(128 + 45 * noise_rng.standard_normal((512, 512, 1)), 0, 255).repeat(3, 2)
                a = float(visible[i]) if calib else float(alpha[i])
                img = Image.fromarray((a * np.asarray(img) + (1 - a) * static).astype(np.uint8))
                name = (f"lens {int(width[i])}s  " if width else "") + f"data {float(cd[i]):.0%}  prior {float(cp[i]):.0%}" if calib else f"agreement {float(agree[i]):.2f}"
            img = label(img, stamp + name, 18)
            if cats is not None:
                img = draw(img, cats.top(tr[i]), dim=max(float(visible[i]), 0.25))
            frame.paste(img, (512 * j, 0))
        writer.append_data(np.asarray(frame))
        n_frames += 1
    writer.close()
    print(path, n_frames, "frames")


if __name__ == "__main__":
    main()
