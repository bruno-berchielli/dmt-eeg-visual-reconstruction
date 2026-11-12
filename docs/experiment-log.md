# Experiment log

Chronological record of what was done, decided and changed, with the reason for each item. Results with numbers are in [results.md](results.md).

## 2026-11-11

### 1. Survey of models and data

- **EEGMirror** (priority candidate in the brief): no published weights. Training requires SEED-DV, SEED and FACED, and SEED-DV requires a license. **Discarded for this round.**
- **ATM** (Li et al., NeurIPS 2024): the authors' HuggingFace repository provides the preprocessed THINGS-EEG2, the CLIP ViT-H/14 features, the *diffusion prior* and precomputed embeddings, but **not the EEG encoder checkpoint**. Generation uses SDXL-Turbo + IP-Adapter ViT-H (`Generation/custom_pipeline.py`).
- **EEG2Video, MindCine:** no weights found; both depend on SEED-DV.
- **DreamDiffusion:** has weights, but was trained on the 128-channel dataset with a block-design confound. Discarded.
- **Dedicated EEG→text (DeWave and similar):** discarded. A published critique shows similar performance with noise as input.

### 2. Verification of the main dataset (Zenodo 3992359)

Confirmed in the files:
- **Recording:** 35 subjects, `.bdf` files at 500 Hz, with 24 EEG channels plus `Status`. The channels are `Fp1 Fp2 Fz F7 F8 FC1 FC2 Cz C3 C4 T7 T8 CPz CP1 CP2 CP5 CP6 M1 M2 Pz P3 P4 O1 O2`.
- **Conditions:** DMT (about 3 to 10 min), eyes closed (EC) and eyes open (EO). The S21 controls are in `.xdf` and were not read.
- **Scales:** `scales_results.csv` has 5D-ASC, NDE, MEQ etc. Subject IDs vary (`S07f`, `s04`).
- **No free-text reports** of what each participant saw. Nothing was invented.
- **No event markers** in the `Status` channel, so the moment of inhalation is not known. The recording was assumed to start close to it.

### 3. Central decision: train the encoder instead of interpolating channels

There is no ready-made EEG→CLIP encoder that accepts this montage. Interpolating 42 of the 63 THINGS channels would be inventing data. **Decision:**
- reimplement the ATM-S architecture (`src/common.py`);
- train it on THINGS-EEG2, which has data from sober participants viewing photos, **using only the channels that the target dataset has**;
- no drug data enters training.

The 21 channels shared with the Zenodo DMT dataset are in `SHARED_CHANNELS`. M1, M2 and Fz do not exist in THINGS.

The training subject was sub-08, the best-performing one in the ATM paper.

### 4. Implementation details and changes

- **THINGS files as pickle:** the `.npy` files on HF are actually Python pickles, and loading them would execute code. For this reason `NumpyOnlyUnpickler` was created (`src/train_encoder.py`), which only reconstructs numpy arrays.
- **Normalization (`normalize_windows`):** average reference, mean removal and per-channel z-score in each window. It is applied identically to THINGS and to the target data.
- **DMT preprocessing** (`src/preprocess_dmt.py`): 0.5 to 45 Hz filter, average reference, resampling to 250 Hz, 1 s windows with 50% overlap, and rejection of windows with peak-to-peak above 200 µV or a flat channel.
- **Discarded subjects:** S05, S14 and S24 were discarded entirely, and S08, S09 and S30 almost entirely. **28 subjects remained** with at least 30 windows of DMT and of EC.
- **Generator** (`src/sdxl.py`): SDXL-Turbo, 4 steps, 512×512, IP-Adapter with scale 1, conditioned only on the CLIP embedding, with no text prompt.
- **Embedding scale:** the THINGS embeddings come normalized (norm 1), and the IP-Adapter encoder produces a norm of about 21.5. For this reason, every vector is normalized and multiplied by 21.5 before generation.
- **Generator validation:** the THINGS features have a cosine of 0.95 to 0.997 with the IP-Adapter encoder, and reconstructions from true features are faithful (`docs/figures/sanity_grid.jpg`).

### 5. Second model: linear decoder

To have a low-capacity "second opinion", a ridge regression was trained on 21 channels × 50 bins of 20 ms, mapping to CLIP (`src/train_linear.py`). λ was chosen on the test set, which makes the metric slightly optimistic; this is acceptable for exploratory purposes.

### 6. First inference: collapse

All conditions (DMT, EC, EO) and even Gaussian noise decode to the same "default guess" of each model: a white egg for ATM and a knife for the linear decoder.

**Method change:** analyze **within-subject contrasts** (DMT − EC), with tests of consistency across subjects against a label permutation (`src/analyze_contrasts.py`).

### 7. Other datasets

- **Ayahuasca:** no open EEG found. OSF uxqcv has only questionnaires.
- **OSF wujgp (IV DMT × placebo):**
  - the `.mat` files have 6 rows per subject, at 1000 Hz, already filtered from 1 to 45 Hz and with artifacts removed by ICA, and they carry no channel labels;
  - the paper says "five central midline electrodes (Oz to FCz)". **Oz, Pz, CPz, Cz, FCz were assumed** for the first 5 rows, which is consistent with the back-to-front alpha gradient;
  - the segments are: 0 = pre-injection, 1 to 3 = post-injection;
  - a 5-channel encoder was trained (`--channels Oz,Pz,CPz,Cz,FCz --tag _osf`).
- **PsiConnect (OpenNeuro ds006110):**
  - 62 participants (2 sessions), with ses-01 as baseline and ses-02 as psilocybin 19 mg;
  - 64-channel actiCAP cap, the same family as THINGS. **62 of the 63 channels** match; only FCz is missing, which is the reference;
  - tasks: rest, guided meditation, music and **cloud movie**;
  - the phenotype data include `CLOUDS.tsv` (patterns, faces and objects seen in the clouds during the EEG) and `ASC11.tsv`;
  - **planned positive control:** sober cloud viewing against rest should make "cloud" rise in the ranking.
  - A 62-channel encoder was trained (`--tag _psi`).

### 8. Bias and specificity tests

The project owner questioned whether "people" was model bias or real similarity. Tests run:
- **Permuted-label contrasts**, to check for generator or encoder bias.
- **Spectrum transplant** (`src/test_spectral_transplant.py`): each subject's EC with the power spectrum of that subject's DMT.
- **Free-text probes** (`src/test_text_probes.py`): "mandala", "fractal", "entity" etc. against neutral objects, using the OpenCLIP ViT-H text encoder.

Point raised by the project owner, accepted: reproducing the effect with the spectrum shows **where** the information is, not that it is irrelevant to the experience. The correct test is **specificity**: different spectral biases should generate different images. This was done in `src/test_specificity.py`.

### 9. Reading and editing the weights

Motivated by the post *Sparks of Weight-Space Programming*:
- **Direct reading of the ATM weights** (`src/read_weights.py` and inspection of the raw values): temporal filters, BatchNorm, `value_embedding` and spatial weights.
- **Gradient-free edits** (`src/weight_surgery.py`, `src/model_variants.py`):
  - output centering;
  - filter pruning;
  - AdaBN with the sober rest of the target dataset.

  The edits were chosen on one half of the THINGS test images and evaluated on the other half.
- **Reading of the IP-Adapter weights:** SVD of `image_proj` and gain per concept category.

### 10. Cloud positive control (PsiConnect)

- **62-channel encoder:** top-1 of 18% and top-5 of 48% on THINGS (best epoch).
- **Fix:** `preprocess_psiconnect.py` wrote the metadata as object arrays. It now writes plain text. The metadata of the existing file were extracted to `data/processed/psiconnect_meta.npz`, read by `decode.load_psiconnect()`.
- **Result:** the control failed in all 4 versions (`cloud_control.py`). The sober cloud, music and meditation tasks, compared with rest, all decode to the same "people and bodies" axis as DMT.
- **Decision:** **do not regenerate the DMT images** with a "winning" version, because none passed the criterion defined before looking at the drug data.

### 11. Imagery control (Figshare 10.6084/m9.figshare.30227503)

- **Motivation:** for DMT there is no "expected image". An imagery dataset with known content tests the seeing → imagining bridge.
- **Search for imagery datasets:** all are small.
  - Figshare (22 participants, 10 images, 32 channels, CC BY 4.0);
  - OpenNeuro ds004306 (13 participants, 3 concepts);
  - ds005815;
  - a preprint with 16 scenes.

  None comes close to the scale of THINGS (1654 concepts).
- **Protocol** (17.1 s per trial): fixation 3 s, image 4 s, mask, black screen with imagery 4 s, "Rest" 4 s.
  - The event marker is the **start of imagery**. This was verified with the occipital evoked responses: −6.0 s (image appears), −2.0 s (image leaves), 0 (black screen), +4.25 s (Rest).
  - The same dataset therefore has **perception and imagery of the same images, in the same participants and with the same equipment**.
- **Channels:** 30 shared with THINGS (Fpz and Fz are left out). A 30-channel encoder was trained (`--tag _vi`), with top-1 of 15.5% and top-5 of 36% on THINGS.
- **Windows** (`preprocess_imagery.py`): perception from −6 to −5 s; imagery from 0.5 to 4 s (6 windows); rest from 4.75 to 7.75 s.
- **Image order:** it was checked and is random (mean run length of 1.43). There is no block-design confound.
- **Open discrepancy:** the authors report 75.8% (animals) and 75.1% (figures) with EEGNet, in a within-subject 80/20 split. The linear classifiers used here do not reproduce this.

### 12. The model depends on the exact instant; the "ignore time" path was tested and abandoned

- **Finding that motivated it:** on the THINGS test, shifting the EEG by 40 ms drops ATM from 13.5% to 3% top-1; with 100 ms it goes to chance. Because the DMT windows are cut at arbitrary instants, the model could only read there what does not depend on time (the spectrum).
- **THINGS-EEG2 protocol confirmed** (paper and data): the photos are shown in rapid sequence, 100 ms on screen and a new one every 200 ms, 20 in a row. The 0 to 1 s window contains the response to 5 photos. 50% of the energy of the mean response is at 5 Hz. Photo-specific information appears at ~70 ms, peaks at ~100 ms and declines until ~500 ms.
- **Test of the ceiling without alignment** (run as a delegated task; main result checked independently): power spectrum, cross-spectrum between channels, bispectrum, wavelet scattering, and networks trained with random shift. Scripts: `shift_features.py`, `test_shift_invariance.py`, `train_shift_robust.py`, `shift_invariance_table.py`.
- **Decision: path abandoned.** Without the instant, about one third of the accuracy remains with an average of 80 repetitions, and practically chance with a single window, which is the DMT situation. Training of the convolutional network was stopped at epoch 17 of 30. The direction becomes **finding the right instant and window and decoding aligned**.
- **Side finding:** evaluating the linear decoder without removing the intercept underestimated the result. With centering, it reaches 17% / 52% among 100 photos. The ATM × linear comparison in section 1 of the results uses different yardsticks and needs to be redone.
- **Other fronts in this round:**
  - Phenomenology predictor (`pheno_features.py`, `pheno_predict.py`): did not work (see results).
  - Per-recording whitening (`align.py`, `gate_perception.py`): marginal effect.
  - Training with more THINGS participants: download cancelled.
- **Literature read in this round:** see `references.md` (imagery × perception, TDLM, registered DMT trial with fMRI, Dream2Image).

### 13. Finding the instant and the window: event-based decoding

Direction set with the project owner: instead of ignoring time, find out **when** and **in which window** to decode.

- **Refactoring by the ATM authors** (`develop` branch, May 2026): it brings validation with early stopping, a diffusion prior, a low-level branch and multi-subject training. The architecture is the same. Adopted here: separate validation and a lean version of the low-level branch. The diffusion prior was left out, because it would make the images more plausible without saying more about the EEG.
- **Window** (`window_sweep.py`): with separate validation, 60 to 500 ms yields the same as the full second.
- **Low-level branch** (`lowlevel.py`): EEG → SDXL internal representation reduced to 4×4 (color and layout), instead of the full 64×64 one used by the authors.
- **Tolerance to timing error** (`tolerance_test.py`, the project owner's suggestion to transform the input): log envelope, smoothing and training with shifted copies. Training with a shift of up to 40 ms was the one chosen (`event_decoders.py`).
- **Event detector** (run as a delegated task, which left `onset_lib.py`, `onset_things.py`, the templates and the continuous data ready; its validation stalled and was redone in a lean form in `detector_check.py` and `lag_curve.py`).
- **Event-based decoding on DMT** (`dmt_events.py`, `dmt_events_render.py`): threshold of each recording calibrated with a surrogate signal with the same spectrum; decoding in the window that starts at each event; subtraction of what the same process returns on the surrogate signal.
- **Deviation from the plan:** DMT was run before the detector check in another laboratory had finished. The check later showed that the detector barely transfers.

### 14. Participants or laboratory? (evening round)

- **Across participants in the same laboratory** (`loso_imagery.py`): detector and decoder trained on 21 participants of the 10-photo dataset work on the left-out participant. Image onset was measured at −5.814 s relative to the marker.
- **10-photo model** (`vi_model.py`, `dmt_events_vi.py`, `dmt_events_vi_render.py`): trained on all 22 participants; does not transfer to THINGS; applied to DMT.
- **More THINGS participants** (`things_multi.py`): participants 01, 02, 03 and 05 were downloaded (41 GB). The decoder is fitted from each participant's sufficient statistics, which allows any subset to be combined without reloading the data.
- **Final DMT images:** generated with the five-participant model (500 ms template) and with the 10-photo model, in `outputs/events/`.

### 15. Time within the recording: videos, selection and drug level

- **Event-based video** (`dmt_events_video.py`): each frame is the mean of the vectors of events close in time (Gaussian window of 30 s, step of 2 s), minus the mean of the phase-randomized surrogate, generated with a fixed seed and no image prior. Generated for S35 and S28. The "mixed" version with color and composition was abandoned: the low-level part comes out almost gray and adds nothing.
- **Excluded recording forced through** (`dmt_events_one.py`): S27 was excluded for having 1.1 min of clean signal under DMT and less than 1 min in the other conditions. Without the minimums, 5 to 15 events per condition come out, and the two models do not agree on any.
- **Chronological chain with image prior** (`dmt_chain.py`): recording split into 20 parts, highest detector peak in each, each image generated from the previous one. Tried and discarded as a method: the sequence locks onto the first composition and only stylizes it. The same vectors without a prior give 20 unrelated scenes.
- **Selection bias discovered:** the artifact filter favors those who had a weak experience. S35, chosen for having the cleanest signal, has the lowest total questionnaire score in the group. The following videos were chosen by questionnaire score (S28).
- **Split-half test** (`dmt_events_video.py … halves`, `halves_group.py`): even and odd events generate two independent trajectories on the same clock. This becomes the yardstick for whether the evolution of a video is in the data.
- **Drug-effect curve** (`drug_curve.py`): the files have no inhalation marker. The state (posterior alpha, slow waves, high frequency, amplitude) was measured in 10 s intervals, relative to the same participant's eyes closed, and compared with a fast-rise, slow-decay curve, with the mirrored curve and with a straight line.
- **Drug-effect direction** (`drug_direction.py`): regression of the decoded vectors on the measured drug level (alpha suppression) and on a muscle indicator, per participant, with windows aligned to the detector peaks and, as a reference, with all windows.
- **Dense decoding** (`dmt_dense.py`): one window at each detector peak, with no threshold (about 350 per recording, against 40 events), only to align the phase. The mean of the phase-randomized surrogate is still subtracted. Videos with this input: `dmt_events_video.py … dense`, time window of 20 s.
- **What the artifact filter rejects:** large-amplitude slow activity, not muscle, and in most cases because of a single bad electrode (T8 in S27, FC2 in S30, FC1 in S14, P4 in S05). Reconstructing the electrode and removing drifts would recover these recordings; this was not done because the participants used in the videos have enough data.
- **Sharpness by agreement** (`… fade`): the moments are split into two halves by 10 s blocks, 20 times; the similarity between the halves at each frame defines how much of the image appears and how much becomes static.
- **State prior** (`… prior`): in low-agreement frames, instead of static, the frame falls back to what the drug level predicts for that participant (regression of that participant's vectors on the measured level). Nothing coming from reports enters generation.
- **Pattern score and stages** (`pattern_score.py`, `stage_kinetics.py`): text probes for geometry, architecture, entities, people, animals, nature and objects, measured along each recording against the drug level, on the real signal, on the phase-randomized surrogate and on the local difference between the two.
- **Stage hypothesis recorded** (the project owner's report, consistent with user reports): very fast geometry at the start, complex architecture next, entities and encounters at the peak, geometry again on the way out. The recordings start with the effect already established, so only the peak and the way out are observable.
- **Calibrated confidence** (`dmt_events_video.py … calibrated`): the previous yardstick ("0.6 = full data") was generous. The vectors live in about 4 effective dimensions, and halves of recordings from different participants reach ±0.45 cosine by chance. Agreement is now measured after whitening with the first 50 principal components and expressed in deviations relative to halves from other participants; the same test applies to the prediction from the drug level (fitted on one half, checked on the other). The image appears according to the larger of the two confidences, and becomes static when neither holds.
- **Category meter** (`categories.py`): bottom strip of each frame with the three categories the vector leans toward and their relative weights (geometry, architecture, entities, people, animals, nature, objects). Video text in English.
- **Stepped lens** (`… lens`): per frame, the narrowest window among 5, 10, 15, 20, 25 and 40 s that passes the criterion. It proved optimistic: with six attempts per frame, narrow windows pass by luck.
- **Inferred lens** (`lens_model.py`, `… inferred`): per instant, the smallest window expected to reach a target reliability, given the local consistency between windows (Spearman-Brown relation). Only choice: the target. Adopted with target 0.65.
- **Configurable step and window** (`step=`, `sigma=`, `target=`): the step defines how many frames per second of recording; playback speed is kept (1 s of video = 12 s of recording).
- **Assumptions revised:** "clean signal" is not the signal of interest, and the detector events do not track the experience (practically the same rate in those who reported a lot and a little). What the DMT EEG carries robustly is the state.

### 16. Possible next steps

- Repair artifacts instead of discarding segments (ASR or ICA), to recover the participants and the moments of strongest experience. First check whether the amplitude filter is rejecting the slow waves of the drug itself.
- Travelling waves (back-to-front direction under DMT, Alamia and Timmermann 2020) as a marker of "when", validated against the questionnaires.
- 24 frames per second version of the inferred-lens video (`step=0.5`), for smoothness only; postponed.
- Scene detection (change points) as an alternative to smoothing, and comparison of the methods by prediction of held-out moments.
- Video with several samples per frame (different seeds of the same vector) and a meter of the beings ↔ structure axis, to make clear that each image is a sample of a tendency.
- **Prediction for an experiment with the inhalation marked:** on the rise, structure (geometry and architecture) should appear before the beings.
- Test the window size (200, 300, 500 ms anchored at onset) and a temporal-token architecture with relative position and attention over the moments.
- Find or obtain a dataset with visual stimuli **and** substance in the same participants, in the same laboratory.
- Calibrate in the target laboratory: the ATM vector contains perception information in another laboratory (+21 points with a local classifier), but misaligned with CLIP. A fit with local visual data could realign it.
- Reproduce the EEGNet of the imagery dataset authors to understand the 75%, checking whether it comes from the first second or from the frontal electrodes (eye movement).
- Make the cloud control pass before interpreting content, for example by fitting the model with visual data from the target laboratory itself.
- `analyze_psiconnect.py` is ready but has not been run yet. With the control failed, it serves only to measure *that* psilocybin changes the state, not *what* is seen.
- Evaluate neutralizing the dominant direction of the IP-Adapter (food ↔ electronics).

### 17. Where things stand

- **Current video method:** windows aligned to all detector peaks, mean of the phase-randomized surrogate subtracted, inferred lens (target 0.65), confidence calibrated against other participants, background prediction from the drug level, static where nothing holds, category meter. Generated for S28 and S29.
- **What the videos are:** materializations of per-category tendencies over time. The category is what the data support; the concrete image comes from the seed and the generator.
- **What is known about the signal:** the path from "beings" at the peak to "structure" on the decline appears in about 20 of 25 participants and comes from the spectrum. The reliable time scale is tens of seconds: a 9-minute recording supports on the order of 6 to 10 distinct segments.
- **Limits:** no reading of individual content; the rise of the effect was not recorded; the participants with the strongest experience have less usable signal (largely because of bad electrodes, which are recoverable).

### 18. Rest contrast, repair and final cases

- **Lens cap** (`max=`): inferred lens capped at 30 s by default. Without the cap, weak recordings open the lens to 60-90 s and the video dwells on few scenes.
- **Rest contrast** (`contrast`): subtracts the subject's eyes-closed mean vector from every window; the confidence floor from other subjects gets the same subtraction.
- **Electrode repair** (`repair_recording.py`): recovers recordings rejected because of one or two bad electrodes; stored as subject `<id>r`. A control on good recordings showed that rebuilding one posterior electrode can invert the category profile, so repaired subjects are kept out of the analysis.
- **Frames streamed to disk**: an interrupted video run leaves a playable partial file.
- **Final cases**: six subjects with strong experiences and unrepaired data (S12, S04, S07, S28, S19, S31), DMT − rest, inferred lens capped at 30 s, one frame per second of recording. Stored in `docs/cases/`.
- **fMRI survey**: the public DMT EEG-fMRI data (`timmer500/DMT_Imaging`) are parcellated (112 regions × 840 volumes per session, EEG as band-power regressors) and cannot feed image reconstruction. Voxel-level public psychedelic fMRI exists for LSD (OpenNeuro ds003059, eyes closed) and psilocybin. Calibration-light or calibration-free decoders exist (Brain-IT, ZEBRA, MindEye2). No published reconstruction under psychedelics was found.

### 19. Design for a follow-up study

1. Same device, more electrodes (dense posterior coverage), for calibration and for the DMT sessions.
2. Sober calibration with paired EEG and images, including complex geometry, visionary art and reported scenes next to everyday objects.
3. Images at unpredictable onsets, so that a model learns "when" as well as "what".
4. A time-resolved report of the experience (minute by minute, or post-hoc image selection per phase) as ground truth.
5. A model with attention over time on top of a pretrained EEG foundation model.
6. First hypothesis to test: the amount of geometric structure in the decoded signal follows the reported intensity of elementary imagery.
