"""Step 0 and detector definitions on THINGS-EEG2 (sub-08).

1. Protocol check: THINGS-EEG2 is RSVP (100 ms on, 100 ms off, 200 ms SOA),
   so every 1 s epoch holds ~5 onsets and the grand average should be
   periodic at 5 Hz. Measured here: share of grand-average power at
   multiples of 5 Hz, correlation between successive 200 ms blocks, and the
   latency at which image-specific signal appears (split-half reliability
   across the 200 test images), which also pins the epoch start to the onset.
2. Matched-filter templates per channel set: the class-independent evoked
   response, as the mean of THINGS crops normalized like the target windows.
   A strictly periodic average only identifies the single-onset response
   folded modulo 200 ms, so the single-onset template is that one RSVP cycle
   (T200, averaged over the five cycles); T500 and T1000 keep 2.5 and 5 cycles.
3. Onset net per channel set (0.5 s windows): positives start at an onset
   (0, 200 or 400 ms into the epoch); negatives start 80-120 ms off an onset
   (anti-phase in the 5 Hz train; multiples of 200 ms are onsets too) or are
   phase-randomized copies of positive windows (same spectrum and channel
   covariance, no time-locking). Evaluated on the held-out test images.

Outputs: models/onset_templates.npz, models/onset_net_<set>.pt,
outputs/onset_things_report.json
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from cache_things import cache_path
from common import MODELS, OUTPUTS, THINGS_CHANNELS, device, normalize_windows
from onset_lib import CHANNEL_SETS, OnsetNet

POSTERIOR = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8", "P7", "P8", "Pz"]
TEMPLATES = {"T200": (50, [0, 50, 100, 150, 200]), "T500": (125, [0, 50, 100]), "T1000": (250, [0])}
NET_L = 125
POS = [0, 50, 100]
NEG_SHIFT = [20, 25, 30, 70, 75, 80, 120, 125]


def phase_randomize(w, rng):
    spec = np.fft.rfft(w, axis=-1)
    ph = np.exp(2j * np.pi * rng.random((w.shape[0], 1, spec.shape[-1])))
    ph[..., 0] = 1
    ph[..., -1] = 1
    return np.fft.irfft(spec * ph, n=w.shape[-1], axis=-1).astype(np.float32)


def protocol_check(test):
    xn = normalize_windows(test.reshape(-1, *test.shape[2:]).astype(np.float32)).reshape(test.shape)
    ga = xn.mean((0, 1))
    p = np.abs(np.fft.rfft(ga, axis=-1)) ** 2
    f = np.fft.rfftfreq(250, 1 / 250)
    blocks = ga.reshape(63, 5, 50).transpose(1, 0, 2).reshape(5, -1)
    a, b = xn[:, 0::2].mean(1), xn[:, 1::2].mean(1)
    a, b = a - a.mean(0), b - b.mean(0)
    rel = ((a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0)))
    post = [THINGS_CHANNELS.index(c) for c in POSTERIOR]
    rp = rel[post].mean(0)
    base = rp[:10].mean()
    first = int(np.argmax(rp > base + 0.5 * (rp.max() - base)))
    return {
        "grand_average_power_share_at_5Hz_multiples": float(p[:, (f % 5 == 0) & (f > 0)].sum() / p[:, f > 0].sum()),
        "top_frequencies_hz": f[np.argsort(p.mean(0))[::-1][:6]].tolist(),
        "corr_successive_200ms_blocks": [float(np.corrcoef(blocks[i], blocks[i + 1])[0, 1]) for i in range(4)],
        "corr_block1_vs_block5": float(np.corrcoef(blocks[0], blocks[4])[0, 1]),
        "image_specific_reliability_posterior_by_20ms": np.round(rp.reshape(50, 5).mean(1), 3).tolist(),
        "image_specific_half_max_latency_ms": first * 4,
        "image_specific_peak_latency_ms": int(np.argmax(rp)) * 4,
    }


def templates(train):
    out = {}
    for name, chans in CHANNEL_SETS.items():
        idx = [THINGS_CHANNELS.index(c) for c in chans]
        for tname, (L, starts) in TEMPLATES.items():
            acc = np.zeros((len(chans), L))
            n = 0
            for i in range(0, len(train), 2000):
                x = train[i:i + 2000][:, :, idx].astype(np.float32).reshape(-1, len(idx), 250)
                for s in starts:
                    acc += normalize_windows(x[..., s:s + L]).sum(0)
                    n += len(x)
            out[f"{name}_{tname}"] = (acc / n).astype(np.float32)
    return out


def make_batch(x, rng):
    """x: (B, C, 250) raw epochs -> windows, labels, kind (0 pos, 1 shift, 2 phase-rand)."""
    B = len(x)
    ps = rng.choice(POS, B)
    pos = np.stack([x[i, :, s:s + NET_L] for i, s in enumerate(ps)])
    ns = rng.choice(NEG_SHIFT, B // 2)
    neg1 = np.stack([x[i, :, s:s + NET_L] for i, s in enumerate(ns)])
    neg2 = phase_randomize(pos[B // 2:], rng)
    w = normalize_windows(np.concatenate([pos, neg1, neg2]))
    y = np.r_[np.ones(B), np.zeros(B)].astype(np.float32)
    kind = np.r_[np.zeros(B), np.ones(B // 2), 2 * np.ones(B - B // 2)]
    return w, y, kind


def train_net(train, test, chans, dev, epochs=6, batch=512, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = [THINGS_CHANNELS.index(c) for c in chans]
    xtr = train[:, :2][:, :, idx].astype(np.float32).reshape(-1, len(idx), 250)
    model = OnsetNet(len(chans), NET_L).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(xtr))
        tot = 0
        for i in range(0, len(perm), batch):
            w, y, _ = make_batch(xtr[perm[i:i + batch]], rng)
            loss = F.binary_cross_entropy_with_logits(model(torch.from_numpy(w).to(dev)), torch.from_numpy(y).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        print(f"  epoch {ep} loss {tot / (2 * len(perm)):.4f}", flush=True)
    model.eval()
    xte = test[:, :20][:, :, idx].astype(np.float32).reshape(-1, len(idx), 250)
    erng = np.random.default_rng(1)
    with torch.no_grad():
        def sc(w):
            return np.concatenate([model(torch.from_numpy(normalize_windows(w[i:i + 4096])).to(dev)).cpu().numpy()
                                   for i in range(0, len(w), 4096)])
        s0 = sc(xte[..., :NET_L])
        res = {}
        for sh in (10, 20, 25, 50):
            s1 = sc(xte[..., sh:sh + NET_L])
            res[f"auc_onset_vs_shift_{sh * 4}ms"] = float(roc_auc_score(np.r_[np.ones(len(s0)), np.zeros(len(s1))], np.r_[s0, s1]))
        s2 = sc(phase_randomize(xte[..., :NET_L], erng))
        res["auc_onset_vs_phase_randomized"] = float(roc_auc_score(np.r_[np.ones(len(s0)), np.zeros(len(s2))], np.r_[s0, s2]))
    return model, res


def main():
    test = np.load(cache_path("test"), mmap_mode="r")
    train = np.load(cache_path("training"), mmap_mode="r")
    report = {"protocol": protocol_check(np.asarray(test))}
    print(json.dumps(report, indent=1), flush=True)
    tpl = templates(train)
    np.savez(MODELS / "onset_templates.npz", **tpl)
    tst = np.asarray(test[:, :20]).astype(np.float32).reshape(-1, 63, 250)
    report["matched_filter_things_test"] = {}
    for key, t in tpl.items():
        name, tname = key.rsplit("_", 1)
        idx = [THINGS_CHANNELS.index(c) for c in CHANNEL_SETS[name]]
        L = t.shape[1]
        if L == 250:
            continue
        def sc(s):
            w = normalize_windows(tst[:, idx, s:s + L])
            return (w * t).sum((1, 2)) / (np.linalg.norm(t) * np.sqrt(t.size))
        s0, s1 = sc(0), sc(25)
        report["matched_filter_things_test"][key] = {
            "auc_onset_vs_shift_100ms": float(roc_auc_score(np.r_[np.ones(len(s0)), np.zeros(len(s1))], np.r_[s0, s1])),
            "mean_ncc_onset": float(s0.mean()), "mean_ncc_shift_100ms": float(s1.mean())}
    dev = device()
    report["onset_net_things_test"] = {}
    for name, chans in CHANNEL_SETS.items():
        print(name, flush=True)
        model, res = train_net(train, test, chans, dev)
        torch.save({"state_dict": model.state_dict(), "channels": chans, "n_times": NET_L}, MODELS / f"onset_net_{name}.pt")
        report["onset_net_things_test"][name] = res
        print(json.dumps(res), flush=True)
    (OUTPUTS / "onset_things_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
