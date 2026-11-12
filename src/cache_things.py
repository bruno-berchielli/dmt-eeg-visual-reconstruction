"""Cache THINGS-EEG2 sub-08 epochs once as float16 on all 63 channels,
without the per-window normalization, so any channel subset can be cut and
normalized later without unpickling the 8.3 GB float64 file again.

Output: data/processed/things_sub-08_{training,test}_63ch_f16.npy
"""
import gc

import numpy as np

from common import DATA
from train_encoder import NumpyOnlyUnpickler


def cache_path(split, sub="08"):
    return DATA / "processed" / f"things_sub-{sub}_{split}_63ch_f16.npy"


def main(sub="08"):
    for split in ("test", "training"):
        out = cache_path(split, sub)
        if out.exists():
            continue
        with open(DATA / "raw/things" / f"sub-{sub}" / f"preprocessed_eeg_{split}.npy", "rb") as f:
            arr = NumpyOnlyUnpickler(f).load()["preprocessed_eeg_data"]
        print(split, arr.shape, arr.dtype, float(np.abs(arr).max()), flush=True)
        np.save(out, arr.astype(np.float16))
        del arr
        gc.collect()


if __name__ == "__main__":
    main()
