# Notice and attribution

## License of this repository

| What | License |
|---|---|
| Code (`src/`) | [MIT](LICENSE) |
| Documentation, reports, figures, videos and GIFs (`README.md`, `docs/`) | [CC BY 4.0](LICENSE-docs), except the third-party material listed below |

Copyright (c) 2026 Bruno Berchielli.

To reuse the videos, figures or text, credit "Bruno Berchielli, DMT EEG Visual Reconstruction", link to this repository, and keep the credit to the source dataset given below.

The repository does not redistribute any dataset or model weights. They are downloaded from their original sources and stay under their own licenses.

## Datasets

| Dataset | Authors | License | How it is used here |
|---|---|---|---|
| [Neural and subjective effects of inhaled DMT in natural settings](https://doi.org/10.5281/zenodo.3992359) (Zenodo 3992359) | C. Pallavicini, F. Cavanna, F. Zambelan, L. A. de la Fuente, M. Arias, C. Romero, R. Carhart-Harris, C. Timmermann, E. Tagliazucchi | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | EEG decoded into the videos, GIFs and figures in `docs/`; questionnaire scores quoted per participant code |
| [THINGS-EEG2](https://osf.io/3jk45/) | A. T. Gifford, K. Dwivedi, G. Roig, R. M. Cichy | CC BY 4.0 | training of the decoders |
| [LidongYang/EEG_Image_decode](https://huggingface.co/datasets/LidongYang/EEG_Image_decode) (THINGS-EEG2 preprocessed, CLIP features) | D. Li et al. | CC BY 4.0 | the files the decoders were trained from |
| [EEG Dataset for Visual Imagery](https://doi.org/10.6084/m9.figshare.30227503) (Figshare 30227503) | J. Gao, L. Zhao | CC BY 4.0 | cross-lab checks |
| [PsiConnect](https://doi.org/10.18112/openneuro.ds006110.v1.2.2) (OpenNeuro ds006110) | L. Novelli et al.; D. Stoliker et al. | CC0 | psilocybin comparison |
| [DMT alters cortical travelling waves](https://osf.io/wujgp/) (OSF wujgp) | A. Alamia, C. Timmermann, D. Nutt, R. VanRullen, R. Carhart-Harris | none stated on the OSF project | intravenous DMT comparison; only summary statistics appear in `docs/reports/` |

Changes made to the Zenodo DMT data: the recordings were filtered, resampled, decoded into image embeddings and rendered as AI-generated pictures. The videos are an illustration of decoded image categories, not a recording of what participants saw, and are not endorsed by the dataset authors.

Papers to cite are listed in [docs/references.md](docs/references.md).

## Models and code

| Component | Authors | License | How it is used here |
|---|---|---|---|
| [ATM, EEG_Image_decode](https://github.com/ncclab-sustech/EEG_Image_decode) | D. Li, C. Wei, S. Li, J. Zou, Q. Liu | MIT, Copyright (c) 2023 DongyangLi | architecture reimplemented in `src/common.py`; training recipe followed in `src/train_encoder.py` |
| [OpenCLIP ViT-H/14 LAION-2B](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | LAION | MIT | embedding space, category probes |
| [IP-Adapter](https://huggingface.co/h94/IP-Adapter) | H. Ye et al., Tencent AI Lab | Apache-2.0 | image-embedding conditioning |
| [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) | Stability AI | [Stability AI Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md) | image generator |

The pictures in the videos, GIFs and figures were generated with SDXL-Turbo. Powered by Stability AI. This Stability AI Model is licensed under the Stability AI Community License, Copyright (c) Stability AI Ltd. All Rights Reserved.

Running the pipeline means using SDXL-Turbo under that license: free for research and non-commercial use, and for commercial use only under the conditions it sets (registration, revenue limit). The MIT license of this code does not change that.

## Third-party material not covered by the CC BY 4.0 grant

- `docs/figures/sanity_grid.jpg`, top row: four stimulus photographs from the THINGS image set (Hebart et al. 2019), shown to compare with their reconstructions. They belong to their original owners.
