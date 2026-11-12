"""Cache continuous, preprocessed recordings for the onset detectors
(imagery lab, Zenodo DMT, OSF) in data/processed/onset_cont/."""
import sys
from concurrent.futures import ProcessPoolExecutor

from common import DATA
from onset_lib import cache_dmt, cache_imagery, cache_osf


def main():
    which = sys.argv[1:] or ["osf", "dmt", "imagery"]
    if "osf" in which:
        cache_osf()
    jobs = []
    if "dmt" in which:
        jobs += [(cache_dmt, p) for p in sorted((DATA / "raw/dmt").glob("*.bdf"))]
    if "imagery" in which:
        jobs += [(cache_imagery, p) for p in sorted((DATA / "raw/imagery/extracted").glob("sub-*/ses-*/eeg/*_eeg.bdf"))]
    with ProcessPoolExecutor(4) as ex:
        for out in [ex.submit(f, p) for f, p in jobs]:
            print(out.result().name, flush=True)


if __name__ == "__main__":
    main()
