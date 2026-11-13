"""Download what the pipeline needs, into the folders the scripts expect.

  python download_data.py videos     trained decoder (2 MB) and the Zenodo DMT
                                     recordings (1.5 GB): enough to make videos
  python download_data.py training   THINGS-EEG2 for the five training subjects
                                     and the CLIP features (52 GB): only to
                                     retrain the decoder with things_multi.py

Files already present with the right size are skipped, so an interrupted
download can be resumed by running the command again. The image generator
(SDXL-Turbo, IP-Adapter) is fetched from Hugging Face on first use.
"""
import json
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import DATA, MODELS

RELEASE = "https://github.com/bruno-berchielli/dmt-eeg-visual-reconstruction/releases/download/decoder-v1/"
DECODER = ("event_ridge_clip_multi.pt", "onset_templates_multi.npz", "category_probes.pt")
ZENODO = "https://zenodo.org/api/records/3992359"
THINGS = "LidongYang/EEG_Image_decode"
SUBJECTS = ("01", "02", "03", "05", "08")
AGENT = {"User-Agent": "dmt-eeg-visual-reconstruction"}


def fetch(url, out, size=None):
    if out.exists() and (size is None or out.stat().st_size == size):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT)) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(out)
    print(f"  {out.name} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


def videos():
    print("Decoder ->", MODELS)
    for name in DECODER:
        fetch(RELEASE + name, MODELS / name)
    print("Zenodo 3992359 ->", DATA / "raw/dmt")
    with urllib.request.urlopen(urllib.request.Request(ZENODO, headers=AGENT)) as r:
        files = json.load(r)["files"]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda f: fetch(f["links"]["self"], DATA / "raw/dmt" / f["key"], f["size"]), files))


def training():
    from huggingface_hub import hf_hub_download
    root = DATA / "raw/things"
    stage = root / "hf"

    def get(name, out):
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(hf_hub_download(THINGS, name, repo_type="dataset", local_dir=stage), out)
        print(f"  {out.relative_to(root)}", flush=True)

    print("THINGS-EEG2 ->", root)
    for split in ("train", "test"):
        get(f"ViT-H-14_features_{split}.pt", root / f"ViT-H-14_features_{split}.pt")
    for sub in SUBJECTS:
        for split in ("training", "test"):
            name = f"preprocessed_eeg_{split}.npy"
            get(f"Preprocessed_data_250Hz/sub-{sub}/{name}", root / f"sub-{sub}" / name)
    shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    what = sys.argv[1:] or ["videos"]
    if not set(what) <= {"videos", "training"}:
        sys.exit(__doc__)
    if "videos" in what:
        videos()
    if "training" in what:
        training()
