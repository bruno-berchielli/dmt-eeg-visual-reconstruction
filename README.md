# DMT EEG Visual Reconstruction

N,N-DMT is widely regarded as the psychedelic with the most vivid and characteristic visuals: dense geometric patterns, impossible architecture, encounters with beings, all within a few minutes and with eyes closed. People describe them in strikingly similar terms, yet the experience leaves no trace anyone else can look at.

This project asks whether scalp EEG can translate part of that subjective visual experience into images, even partially. It takes EEG recorded while participants were under DMT, decodes it with models trained on sober people viewing images (THINGS-EEG2), and renders the result over time with a diffusion generator. Each video below is one participant's DMT session minus their own eyes-closed rest.

<table>
  <tr>
    <td align="center" width="33%"><img src="docs/cases/S12.gif" width="270" alt="S12"><br><b>S12</b><br>reported patterns 90 · scenes 73<br><a href="docs/cases/S12_dmt_minus_rest.mp4?raw=true">download video (mp4)</a></td>
    <td align="center" width="33%"><img src="docs/cases/S04.gif" width="270" alt="S04"><br><b>S04</b><br>reported patterns 85 · scenes 72<br><a href="docs/cases/S04_dmt_minus_rest.mp4?raw=true">download video (mp4)</a></td>
    <td align="center" width="33%"><img src="docs/cases/S07.gif" width="270" alt="S07"><br><b>S07</b><br>reported patterns 83 · scenes 33<br><a href="docs/cases/S07_dmt_minus_rest.mp4?raw=true">download video (mp4)</a></td>
  </tr>
  <tr>
    <td align="center" width="33%"><img src="docs/cases/S28.gif" width="270" alt="S28"><br><b>S28</b><br>reported patterns 81 · scenes 59<br><a href="docs/cases/S28_dmt_minus_rest.mp4?raw=true">download video (mp4)</a></td>
    <td align="center" width="33%"><img src="docs/cases/S19.gif" width="270" alt="S19"><br><b>S19</b><br>reported patterns 87 · scenes 48<br><a href="docs/cases/S19_dmt_minus_rest.mp4?raw=true">download video (mp4)</a></td>
    <td align="center" width="33%"><img src="docs/cases/S31.gif" width="270" alt="S31"><br><b>S31</b><br>reported patterns 84 · scenes 57<br><a href="docs/cases/S31_dmt_minus_rest.mp4?raw=true">download video (mp4)</a></td>
  </tr>
</table>

Under each video: what that participant reported seeing during the session, on a 0-100 questionnaire scale (5D-ASC). *Patterns* is simple imagery such as colors, shapes and geometric patterns; *scenes* is complex imagery such as scenes, beings and objects. These six are among the strongest experiences in the dataset.

Top strip: time in the recording, time-window width (lens), confidence in the local data and in the state prediction. Bottom strip: the three image categories the decoded vector leans to. Static means the data do not support a picture at that moment. Full videos and per-participant details: [docs/cases](docs/cases/README.md).

## What it does

1. **Decode.** A ridge decoder trained on five THINGS-EEG2 participants maps 1 s EEG windows to CLIP ViT-H/14 image embeddings. Windows are aligned to the peaks of an image-onset template, about 350 per recording.
2. **Stabilize.** A phase-randomized surrogate of each recording gives the decoder's response to the spectrum alone, which is subtracted. The participant's eyes-closed recording gives the rest baseline.
3. **Infer the lens.** For every moment, the smallest time window whose average is expected to be reliable, from the local consistency of the decoded windows (Spearman-Brown), capped at 30 s.
4. **Measure confidence.** Split-half agreement in a whitened space, expressed against recordings of other participants. A state prediction from the measured alpha suppression fills in where local data are weak; static appears where neither holds.
5. **Render.** SDXL-Turbo with IP-Adapter, fixed seed, no text prompt and no image prior: the picture changes only when the decoded vector changes. A category readout from text probes is drawn on every frame.

Runs on a laptop GPU (Apple M4 Pro, about 2 s per frame).

## Results so far

- **The decoded trajectory is a property of the recording.** Independent halves of the data agree over time, and the direction in which vectors move with the drug level is shared across participants (consistency p = 0.001, n = 25).
- **A common course.** Along the session the readout moves between "beings" (people, animals, entities) and "structure" (geometry, architecture) in about 20 of 25 participants, tracking the measured alpha suppression.
- **Link to reports.** Participants who reported stronger elementary imagery show more geometric structure in the decoded signal (Spearman 0.38, p = 0.06, n = 25). S07, who reported patterns with few scenes, yields an almost purely geometric video.
- **Transfer across people works within a lab**: a detector and decoder trained on 21 participants of a single-image dataset work on the 22nd (AUC 0.71, 22.8% single-trial accuracy at 10% chance).
- **Reliable time scale: tens of seconds.** The lens model predicts the reliability it delivers (target 0.5, obtained 0.52-0.59).
- **Recovering rejected sessions.** Most sessions lost to artifact rejection fail on a single electrode; rebuilding it recovers up to 97% of the recording.

Next step: calibration on the same device (sober image viewing, including geometric and visionary imagery, with unpredictable onsets) to move from state readout toward content. Design in the [experiment log](docs/experiment-log.md).

- [Experiment log](docs/experiment-log.md): everything tried, in order
- [Results](docs/results.md): numbers and controls
- [References](docs/references.md): papers, datasets, model repos
- [Original brief](docs/brief.md)

## Models and generator

| Component | Source |
|---|---|
| ATM (EEG encoder, reimplemented as ATM-S with channel subsets) | [ncclab-sustech/EEG_Image_decode](https://github.com/ncclab-sustech/EEG_Image_decode), [arXiv 2403.07721](https://arxiv.org/abs/2403.07721) |
| Image embedding space | [OpenCLIP ViT-H/14 LAION-2B](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) |
| Generator | [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo), 4 steps, 512 px, no text prompt |
| Conditioning | [IP-Adapter](https://huggingface.co/h94/IP-Adapter) `ip-adapter_sdxl_vit-h` |

## Data

| Dataset | Content | Path |
|---|---|---|
| [Zenodo 3992359](https://zenodo.org/records/3992359) | inhaled DMT, 35 participants, 24 channels, 500 Hz, DMT / eyes closed / eyes open, 5D-ASC | `data/raw/dmt/` |
| [OSF wujgp](https://osf.io/wujgp/) | intravenous DMT vs placebo, 12 participants, 6 midline electrodes | `data/raw/osf_waves/` |
| [OpenNeuro ds006110](https://openneuro.org/datasets/ds006110) (PsiConnect) | psilocybin vs baseline, 64 channels, cloud movie | `data/raw/psiconnect/` |
| [Figshare 30227503](https://doi.org/10.6084/m9.figshare.30227503) | perception and imagery of 10 images, 22 participants, 32 channels | `data/raw/imagery/` |
| [HF LidongYang/EEG_Image_decode](https://huggingface.co/datasets/LidongYang/EEG_Image_decode) | THINGS-EEG2 preprocessed at 250 Hz (sub-01, 02, 03, 05, 08), CLIP features, images | `data/raw/things/` |

Data, trained models and outputs are not versioned (about 80 GB). Reports are copied to `docs/reports/`.

## Setup

```
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision mne pyxdf numpy scipy pandas scikit-learn matplotlib diffusers transformers \
  accelerate safetensors open_clip_torch einops pillow imageio imageio-ffmpeg huggingface_hub h5py
```

Tested on an Apple M4 Pro (48 GB) with MPS: about 2 s per generated frame.

## Training

Run from `src/`. All decoders map EEG windows (250 Hz, 1 s, average reference, per-channel z-score) to CLIP ViT-H/14 image embeddings of the THINGS images.

| Model | Script | Notes |
|---|---|---|
| ATM-S encoder, one subject, any channel subset | `train_encoder.py --channels … --tag …` | contrastive + MSE, as in the ATM repo; retrained because published weights need 63 channels |
| Linear baseline | `train_linear.py` | ridge on the waveform |
| Jitter-tolerant event decoder | `event_decoders.py` | centered ridge, 20 ms bins from 100 to 540 ms, copies shifted ±40 ms |
| Five-subject decoder and onset template (used by the videos) | `things_multi.py 08 01 02 03 05` | ridge from per-subject sufficient statistics; matched-filter template on 14 channels |
| Single-image lab model | `vi_model.py` | detector and decoder trained on the 10-image dataset |

## Pipeline

| Stage | Scripts |
|---|---|
| Preprocess | `preprocess_dmt.py`, `preprocess_osf.py`, `preprocess_psiconnect.py`, `preprocess_imagery.py`, `onset_cache.py`, `cache_things.py` |
| First-phase decoding and contrasts | `embed_dmt.py`, `analyze_contrasts.py`, `render.py`, `analyze_osf.py` |
| Bias and specificity controls | `test_spectral_transplant.py`, `test_specificity.py`, `test_text_probes.py`, `cloud_control.py`, `analyze_imagery.py`, `test_imagery_waveform.py`, `gate_perception.py` |
| Weight reading and gradient-free edits | `read_weights.py`, `weight_surgery.py`, `model_variants.py` |
| Time dependence (abandoned shift-invariant path) | `test_shift_invariance.py`, `shift_features.py`, `train_shift_robust.py`, `shift_invariance_table.py` |
| Window, low-level target, tolerance | `window_sweep.py`, `lowlevel.py`, `tolerance_test.py` |
| Onset detection and cross-lab checks | `onset_lib.py`, `onset_things.py`, `onset_imagery.py`, `detector_check.py`, `lag_curve.py`, `loso_imagery.py` |
| Event decoding on DMT | `dmt_events.py`, `dmt_events_render.py`, `dmt_events_vi.py`, `dmt_events_vi_render.py`, `dmt_events_one.py`, `dmt_chain.py` |
| Dense decoding, reliability, lens | `dmt_dense.py`, `halves_group.py`, `lens_model.py` |
| Drug level, direction, stages | `drug_curve.py`, `drug_direction.py`, `pattern_score.py`, `stage_kinetics.py` |
| Videos | `dmt_events_video.py`, `categories.py`, `repair_recording.py` |
| Phenomenology prediction | `pheno_features.py`, `pheno_predict.py`, `analyze_psiconnect.py` |

Shared modules: `common.py` (channels, normalization, architecture), `decode.py` (embedding, concept space, consistency tests), `sdxl.py` (generator).

Final videos:

```
cd src
python onset_cache.py dmt && python things_multi.py 08 01 02 03 05
python drug_curve.py && python dmt_dense.py
python dmt_events_video.py S07 T500multi DMT inferred contrast step=1 target=0.65 max=30
```

## Layout

```
src/            scripts
docs/           brief, experiment log, results, references, reports, figures, cases
data/ models/ outputs/ external/    not versioned
```
