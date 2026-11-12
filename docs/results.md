# Results

This is the detailed record: every measurement and control experiment, in the order the work was done. For an overview, read the [README](../README.md) first. Terms and abbreviations are defined in the [glossary](glossary.md). Early sections describe approaches that were later replaced; section 13 covers the method behind the published videos, and section 14 summarizes what holds.

Status as of 2026-11-11. The numbers come from the reports in [`docs/reports/`](reports/), and the figures are in [`docs/figures/`](figures/). Chance on THINGS: 0.5% top-1 and 2.5% top-5 with 200 alternatives; 1% and 5% with 100.

## 1. Encoders trained on THINGS-EEG2 (sub-08)

Performance on the THINGS test set, with 200 alternatives and repetitions averaged:

| Model | Channels | Top-1 | Top-5 | Used in |
|---|---|---|---|---|
| ATM-S | 21 (shared with Zenodo) | 13.5% | 29.5% | DMT Zenodo |
| Linear (ridge) | 21 | 4.0% | 12.5% | DMT Zenodo |
| ATM-S | 5 (Oz, Pz, CPz, Cz, FCz) | 1.5% | 9.5% | DMT OSF |
| ATM-S | 62 (all except FCz) | 18% | 48% | PsiConnect |

For comparison, the ATM paper reports top-1 of about 28% and top-5 of about 60% with 63 channels. The performance loss with 21 channels comes mostly from the missing occipito-parietal electrodes (Oz, POz, PO3/4/7/8, P5 to P8).

## 2. Inhaled DMT (Zenodo), 28 subjects

### 2.1 Raw output: collapse

For all conditions and for Gaussian noise, each model returns the same "default guess":
- ATM: "cup, hook, pendulum", which becomes a white egg in the image;
- linear: "fence, gutter, rolling pin", which becomes a knife.

The mean cosine between the DMT means of different subjects is **0.999** (`nearest_concepts.csv`).

### 2.2 Contrast DMT − eyes closed (EC)

Source: `contrast_report.json`. Permutations shuffle the labels within subject (200 permutations).

| | ATM | Linear |
|---|---|---|
| Consistency across subjects (mean pairwise cosine) | **0.288** (null ≈ 0, p = 0.005) | 0.013 (p = 0.11) |
| Cosine with the EO − EC effect | 0.86 | 0.73 |
| Nearest concepts | arm, scalpel, blindfold, nail file, stilt, harness, toga, costume | punch, juicer, footbath, sheet, cream, closet |
| Spearman with 5D-ComplexImagery | −0.21 (p = 0.28) | 0.19 (p = 0.34) |
| Spearman with 5D-ElementaryImagery | −0.15 (p = 0.45) | −0.40 (p = 0.034, uncorrected) |

- **The effect is consistent in ATM, but 86% similar to opening the eyes.**
- **After removing the EO direction** (leave-one-subject-out): in ATM a consistent residual remains (0.103, p = 0.005) that points to **pantsuit, toga, costume, woman, mannequin, girl, face, doll**. In the linear model the residual is not consistent (p = 0.54), but the nearest concept is similar (costume, mannequin, dollhouse). The cosine between the residuals of the two models is 0.31.
- **Visually**, ATM generates full-body people, facing forward, many holding long objects (`figures/peek_dmt_ec.jpg`). Also recurring are grid structures with colored elements (S27 in both models), dense fields of small colored objects, and vegetation or organic textures, the latter in the linear model.
- **The videos** (20 s sliding window, 5 s step) alternate between the same families: people, plain paper and tools (`figures/peek_video_S12.jpg`).

### 2.3 Is the bias in the generator?

No:
- contrasts with permuted labels decode to random things (poles, deer, amplifiers), with a consistency of −0.002;
- the IP-Adapter weights amplify people (6.1), nature/texture (6.2), patterns (5.9) and objects (5.6) almost equally (section 5).

### 2.4 Spectral transplant

Source: `spectral_transplant_report.json`. Each subject's eyes-closed EEG receives that subject's DMT power spectrum, with phases preserved.

| | ATM | Linear |
|---|---|---|
| Group cosine: real × transplant | **0.86** | 0.55 |
| Per-subject cosine (median) | 0.74 | 0.29 |
| Transplant concepts | blowgun, stilt, arm, man, costume, boy, girl, woman | shaker, pitcher, watering can, vase |
| Non-spectral residual (consistency) | 0.171: wax paper, paper, canvas, tape | 0.013 (not consistent) |

**In ATM, the "people" come from the change in spectral power alone.** In the linear model, the image from real DMT (people in a room) is **not** reproduced by the transplant (which gives bottles).

### 2.5 Specificity: different spectral biases

Source: `specificity_report.json`.

| Bias applied to eyes-closed EEG | ATM decodes | Cosine with DMT (ATM) |
|---|---|---|
| Alpha decrease only (8 to 13 Hz × 0.5) | harness, stilt, arm, costume (people) | 0.79 |
| Eyes open (real) | nail file, scalpel, letter opener, arm | 0.86 |
| Theta increase only | wire cutters, combination lock, pliers | 0.08 |
| Fast-activity increase only (20 to 45 Hz × 2) | bedpan, pot, bowl, ashtray (containers); in the direct images, landscapes | −0.46 |
| Flattened spectrum | bedpan, pot, bowl, jug | −0.68 |
| Muscle artifact | bedpan, inkwell, bowl, mallet | −0.46 |
| DMT spectrum | blowgun, stilt, arm, man | 0.86 |

- **ATM works as a rhythm-to-image dictionary**: different biases generate different categories. The DMT signature lands on the same "word" as the alpha decrease.
- **The muscle artifact goes in the opposite direction**, which rules out muscle tension as the cause of the "people".
- **The linear model is not very specific**: all biases have cosines of 0.4 to 0.9 with each other.

### 2.6 Free-text probes

Source: `text_probe_report.json`. Each category is measured against neutral objects; ATM.

| Category | DMT − EC | Compared with EO − EC | Compared with the transplant |
|---|---|---|---|
| Geometry (mandala, fractal, kaleidoscope…) | **−0.020** (p = 0.04) | equal | lower than the transplant (+0.024) |
| Entities (alien entity, machine elf, divine being…) | +0.017 (p = 0.03) | higher (p = 0.02) | lower than the transplant (+0.041) |
| People | +0.070 (p = 0.0002) | higher (p = 0.007) | lower than the transplant (+0.122) |

In the linear model, nothing is significant. There are multiple comparisons; only "people" is robust. **There is no geometry signal**, and "entities" follows the "people" axis, which the spectrum alone reproduces more strongly.

## 3. Intravenous DMT × placebo (OSF), 12 subjects, eyes closed

Source: `osf_report.json`. The contrast is (DMT post − DMT pre) − (placebo post − placebo pre); 5-channel encoder.

| Post-injection segment | Consistency | p | Concepts |
|---|---|---|---|
| 1 (peak) | **0.213** | **0.002** | mulch, moss, gravel, rock, sequin, jigsaw puzzle, hay, gem, turf, bark |
| 2 | 0.108 | 0.048 | barnacle, moss, worm, pinecone, fungus, bark |
| 3 | 0.008 | 0.35 | nothing consistent |

- **The time course follows the pharmacokinetics of intravenous DMT**, with placebo subtracted.
- **The content is dense, grainy, organic textures.** "Face" ranks 1398 out of 1654, so there are no "people" here.
- Caveat: the 5-channel encoder is weak, and the channel identity is an assumption (section 4 of the [log](log.md)).

## 4. Reading and editing the ATM weights (21 channels)

Sources: `weights_reading_encoder_sub-08.json` and `weight_surgery_report.json`.

### What the weights show

1. **Heaviest electrodes:** O1 (1.00), O2 (0.93), Pz, P3, P4 (about 0.85). The model learned to prioritize the visual cortex.
2. **The ATM "temporal" convolution slides over an axis with no temporal order**, which is the output of `value_embedding` (250×250). The mean correlation between neighboring rows of that axis is 0.09, and the filters look like noise. The filters that carry variance are the ones with a high "sum", which average blocks of features: the correlation between |sum| and variance is **0.94**. About 10 of the 40 filters dominate.
3. **BatchNorm amplifies the low-variance filters by up to 79×** (filter 9).
4. **The main input directions of `value_embedding` peak at 18, 12, 13 and 15 Hz**, at the boundary between alpha and beta. This explains the sensitivity to the alpha decrease.
5. **No effective filter is dominated by alpha** (30 by beta, 9 by delta, 1 by theta). The alpha decrease acts indirectly, through the per-window normalization, which raises the proportion of delta and beta.

### Gradient-free edits

Chosen on one half of the THINGS test set and evaluated on the other half (100 alternatives):

| Version | Top-1 | Top-5 | Top-1 single segment | Top-5 single segment |
|---|---|---|---|---|
| Original | 19% | 47% | 3% | 17% |
| + output centering | **23%** | **54%** | **6%** | **24%** |
| + centering + 6 filters pruned | 21% | 58% | 5% | 23% |

With the full 200 alternatives:
- **Pruning the 10 highest-variance filters** (the "rhythm pathway") **keeps or improves** the result: top-1 from 13.5% to 13.5% and top-5 from 29.5% to 32.5%.
- **Pruning the high-gain filters** (the "detail pathway") **makes it worse**: with 24 filters cut, top-1 drops to 8%. The hypothesis that they were noise was **refuted**.

### Effect on DMT

| Version | Mean cosine across subjects (raw DMT) | Subjects with a distinct top-3 | DMT − EC consistency | Cosine with the transplant |
|---|---|---|---|---|
| Original | 0.999 | 5 of 28 | 0.288 | 0.86 |
| Centered | 0.987 | 5 of 28 | 0.295 | — |
| Centered + AdaBN (Zenodo EC) | **0.565** | **28 of 28** | 0.293 | 0.88 |
| Detail pathway only | — | — | 0.234 (p = 0.001) | 0.81 |

- **AdaBN undoes the collapse**, but the DMT effect still points to bodies, people and tools.
- **In the detail pathway**, the non-spectral residual is consistent (0.158, p = 0.001) and points to **wax paper, paper, tape, plastic film, gauze, canvas**. It does not correlate with the imagery scales (ρ ≈ 0), and face, kaleidoscope and stained glass are in the bottom half of the ranking.

## 5. IP-Adapter weights (CLIP → SDXL bridge)

- **Size:** 349 M parameters. The `image_proj` layer (8192×1024) has a dominant singular value of **56.5**, against 14.6 for the second.
- **Dominant direction:** food and drink (wedding cake, cheeseburger, margarita) ↔ screens and electronics (screen, microphone). It is a candidate explanation for the excess of food in the linear model's images (hypothesis not yet tested).
- **Gain per category:** people 6.1, nature 6.2, patterns 5.9, objects 5.6. Random direction: 4.8 ± 0.4. Real photos: 6.0.
- **The ATM DMT − EC contrast has a gain of 8.6**, aligned with directions the generator "hears" well. This is why the contrast images come out sharp, and that sharpness overstates the strength of the signal in the EEG.

## 6. Positive control: clouds in PsiConnect (sober participants)

Source: `cloud_control_report.json`. Per-subject contrast: cloud movie − rest, in ses-01 (sober), with the 62-channel encoder.

| Model version | Rank of "cloud" in the group (out of 1654) | Per-subject median | Consistency | Concepts |
|---|---|---|---|---|
| original | 775 | 824 | 0.57 | arm, stilt, leg, baton, hula hoop, mannequin |
| detail | 975 | 931 | 0.49 | arm, stilt, leg, mannequin |
| AdaBN | 858 | 913 | 0.56 | arm, stilt, leg, harness |
| detail + AdaBN | 1120 | 1022 | 0.50 | arm, stilt, leg, pogo stick |

**No version decodes cloud.** No subject has "cloud" among the first 100 concepts.

Other sober tasks, compared with rest (original version):

| Contrast | Cosine with DMT − EC (ATM, Zenodo) | Concepts |
|---|---|---|
| clouds − rest | 0.66 | arm, stilt, leg, hula hoop |
| music − rest | 0.65 | arm, hip, hula hoop, blowgun |
| meditation − rest | 0.55 | arm, blowgun, stilt, harness |

The cosine between clouds and music is **0.94**.

**The "people and bodies" axis is the decoders' generic response to "leaving rest"**, and it appears in another lab, with other equipment and another encoder, for any task, including an auditory one.

## 7. Imagery control: seeing × imagining the same 10 images (22 participants)

Sources: `imagery_report.json` and `imagery_waveform_report.json`. The 30-channel encoder was trained on THINGS; the 10 images are circle, square, pentagram, dog, fish, bird, cup, chair, clock and scissors.

**THINGS model applied directly**, comparing the decoded vector with the CLIP embedding of the 10 real photos. Results with AdaBN are equivalent.

| | Accuracy among 10 (chance 10%) | Within category (chance ~30%) | Mean rank (chance 5.5) | p |
|---|---|---|---|---|
| Perception | 10.9% | 43% | 5.19 | 0.07 |
| Imagery | 12.7% | 29.5% | 5.47 | 0.56 |

**Within-lab ceiling**: logistic regression with cross-validation per subject and task, reported as gain over chance.

| Features | Perception | Imagery |
|---|---|---|
| Band power (normalized windows) | +11 points (p = 7×10⁻¹²) | +1.2 points (p = 0.02) |
| ATM vector (1024-d) | **+21 points** (p = 9×10⁻¹³) | +0.3 points (p = 0.35) |
| Waveform 0.5–20 Hz (ses-01, 66 recordings) | **+14.5 points** (p = 2×10⁻¹²) | 0–4 s: +1.7 points (p = 0.009); 1–4 s: +0.8 points (p = 0.09) |
| Waveform, Fp1, Fp2, F7, F8 only | — | 0–4 s: **+1.7 points** (p = 0.003) |

- **Transfer between labs fails even for perception.** But the ATM vector carries information about what is seen (+21 points with a local classifier). That is, the representation exists, but it is misaligned with CLIP in another lab.
- **Imagery barely shows up in the EEG.** The little that shows up is in the first second and is the same using only the frontal electrodes, which is compatible with eye movement or with the screen transition, not with mental content.
- **Open discrepancy:** the authors report about 75% (3 alternatives) with EEGNet. This was not reproduced here.

## 8. Time dependence and ceiling without alignment (THINGS)

**Shifting the EEG in time** (21-channel ATM, 200 alternatives):

| Shift | Top-1 | Top-5 |
|---|---|---|
| 0 ms | 13.5% | 29.5% |
| 40 ms | 3% | 16% |
| 100 ms | 2% | 5.5% |
| 200 ms | 0% | 0% |
| random per repetition | 0% | 3% |

**Representations that do not depend on timing.** Source: `shift_invariance_report.json`. 100 alternatives (odd test images; settings chosen on the even ones); chance of 1% / 5%. "Mean" uses the 80 repetitions; "single" is one window only.

| Representation | Aligned, mean | Aligned, single | Shifted, mean | Shifted, single |
|---|---|---|---|---|
| Waveform + ridge | 17 / 52 | 5.0 / 18.3 | 1 / 6 | 1.1 / 5.4 |
| Power spectrum | 2 / 14 | 1.1 / 6.0 | same | same |
| Cross-spectrum between channels | 2 / 11 | 1.3 / 6.1 | same | same |
| Bispectrum | 5 / 12 | 1.1 / 5.4 | same | same |
| Wavelet scattering | 2 / 14 | 1.25 / 6.0 | same | same |
| All invariants together | 4 / 18 | 1.4 / 6.6 | 4 / 18 | 1.4 / 6.6 |
| ATM trained with shifts | chance | — | chance | — |
| Invariant convolutional network (stopped at epoch 17 of 30) | 5 / 24 | — | 5 / 24 | — |

- The "all invariants" result was reproduced independently with another draw of shifts (4% / 18%).
- The convolutional network numbers are from the last recorded epoch, with no epoch selection on a held-out set; treat them as approximate.
- Because of the rapid photo sequence (5 Hz rhythm), even the numbers without alignment are an upper bound for windows cut at random.
- **Conclusion:** most of the information about the image is in the exact shape of the response over time. Without the timing, a single window is practically at chance.

## 9. Phenomenology predictor (EEG → questionnaire scores)

Source: `pheno_report.json`. Features: power change per band and channel (drug − baseline), on the 21 shared channels. Training on psilocybin (PsiConnect, 56 participants), test on DMT (Zenodo, 28 participants).

| Scale | Within psilocybin (leave-one-participant-out) | Psilocybin → DMT |
|---|---|---|
| Elementary imagery | ρ = 0.12 (p = 0.19) | ρ = −0.10 |
| Complex imagery | ρ = 0.10 (p = 0.24) | ρ = −0.24 |
| Audiovisual synesthesia | **ρ = 0.39 (p = 0.002)** | ρ = 0.01 |

The imagery scores are not predictable from band power. Audiovisual synesthesia is predictable within psilocybin, but does not transfer to DMT. No image was generated through this route.

## 10. When, how much and what (within THINGS)

Sources: `window_sweep_report.json`, `lowlevel_report.json`, `tolerance_report.json`, `event_decoders_report.json`. 200 alternatives, 80 repetitions; chance of 0.5% / 2.5%.

**Window** (centered linear decoder, chosen on validation):

| Window after onset | Validation top-5 | Test top-1 / top-5 |
|---|---|---|
| 0–200 ms | 13% | 3.5 / 18.5 |
| 0–500 ms | 20% | 11.5 / 36 |
| 60–500 ms | 21.5% | 12 / 35.5 |
| 0–1000 ms | 18.5% | 12.5 / 35 |
| 500–1000 ms | 7% | 6 / 14 |

**Tolerance to timing error** (window from 100 to 540 ms):

| Input | Aligned | ±20 ms | Random error up to ±40 ms |
|---|---|---|---|
| Waveform, 20 ms bins | 12.5 / 35.5 | ~1.5 / 6 | 2 / 7.5 |
| Waveform, 60 ms bins | 8.5 / 28 | 3.5–8 / 12–20 | 5 / 24.5 |
| Log of the band envelope | 2.5 / 10 | same | 3 / 9.5 |
| Trained with shifts of up to ±40 ms | 8 / 25.5 | ~9 / 20–25 | **8 / 28** |

**Low level** (color and layout, 4×4 grid): correlation of 0.39 with the real pattern and top-5 of 15.5% when aligned; with tolerated shift, 0.18 to 0.23 and 6.5 to 9.5%.

## 11. Transfer: participants × lab

| Test | Result | Chance |
|---|---|---|
| THINGS → 10 photos, single-participant decoder, at the measured onset | mean rank 5.80 | 5.5 |
| THINGS → 10 photos, best timing among 21 tested | 4.84 (one participant); 4.54 (five participants) | 5.5 |
| THINGS → 10 photos, detector (200 ms / 500 ms template) | rise of 0.17 / 0.40 standard deviations when the image appears | 0 |
| 10 photos, training on 21 participants and test on the 22nd: detector | AUC 0.71 (all above 0.57) | 0.50 |
| 10 photos, same: decoder, **single trial** | 22.8% among 10; 43.6% within category | 10%; 30% |
| 10 photos → THINGS: detector | AUC 0.50 | 0.50 |
| 10 photos → THINGS: content (7 concepts in common) | 15.7% | 14.3% |
| THINGS, training on 4 participants and test on another (participant 08) | 3–4% / 13.5–15.5% | 0.5% / 2.5% |
| THINGS, participant 08 only, on itself | 9% / 26.5% | same |

Sources: `lag_curve.json`, `detector_check_z14.json`, `loso_imagery_report.json`, `vi_model_report.json`, `things_multi_report.json`.

- **Across participants, in the same lab and protocol, transfer works**, both on the 10 photos and on THINGS.
- **Across labs and protocols, it does not work in either direction.** The main obstacle is not the difference between brains, but the protocol (photos in rapid sequence versus one image at a time) and the lab.
- The THINGS → 10 photos curve oscillates and flips sign every ~100 ms near image onset, which indicates that the decoder reacts to the visual response, but with the 5 Hz rhythm learned in training.

## 12. Event-based decoding on DMT (Zenodo)

Sources: `dmt_events_report_T200.json`, `dmt_events_report_T500multi.json`, `dmt_events_vi_report.json`.

| Model | Events/min: DMT, EC, EO | On the phase-randomized surrogate | DMT × EC (excess) | Content consistent across participants |
|---|---|---|---|---|
| THINGS, one participant, 200 ms template | 10.3 / 10.5 / 9.4 | 12.0 / 12.3 / 13.3 | p = 0.71 | DMT no (p = 0.46); EC no; **EO yes (p = 0.005)**; DMT − EC no (p = 0.83) |
| THINGS, five participants, 500 ms template | 9.1 / 9.5 / 8.1 | 10.2 / 10.7 / 10.3 | — | DMT no (p = 0.16); **EC yes (p = 0.012)**; EO no (p = 0.25); DMT − EC no (p = 0.34) |
| 10 photos, 22 participants | 3.7 / 5.4 / 5.6 | 4.5 / 6.3 / 5.7 | p = 0.60 | see below |

- **In no model does DMT have more events than eyes closed**, and in all of them there are fewer events than in the phase-randomized surrogate with the same spectrum.
- **The content in the DMT events is not consistent across participants**, nor is the difference from eyes closed. The condition that shows up as consistent changes from one model to the other (EO in the first, EC in the second), so the eyes-open finding did not hold up.
- **Individual signature:** half of a participant's events identifies the other half 4 to 5 times above chance, equally in DMT and in eyes closed.
- **10-photo model:** the per-category vote on DMT leans toward geometric figures (above zero, p = 0.03; eyes closed p = 0.75), with circle as the most voted image. The paired DMT × EC comparison is not significant (p = 0.25, n = 15).
- **Group images with the five-participant model:** DMT, EC and EO give brown-furred animals in vegetation; the DMT − EC contrast gives repetitive colored patterns (citrus fruit slices, striped straws), with no statistical consistency across participants.

## 13. Time within the recording: selection, halves, curve and effect direction

Sources: `reports/halves_group_T500multi.json`, `reports/halves_group_T200.json`, `reports/drug_curve.json`, `reports/drug_direction.json`. Figures: `figures/chain_S35_prior.jpg`, `figures/chain_S35_no_prior.jpg`.

### Duration and selection

| Condition | Total duration (median; range) | Clean signal (median) |
|---|---|---|
| DMT | 6.4 min (2.6 to 13.8) | 3.4 min |
| Eyes closed | 5.0 min (2.4 to 10.0) | 4.1 min |
| Eyes open | 5.0 min (2.6 to 5.3) | 2.3 min |

- **The stronger the experience, the less clean signal** (ρ = −0.33 between the 5D-ASC total score and the clean minutes, 35 participants). Of the eight participants with the highest scores, three have no clean signal at all on DMT and one has 1.1 min.
- **S35, with the cleanest signal (7.4 of 7.5 min), has the lowest total score in the group** (325; median 834) and the lowest complex imagery score (10; median 53).
- **The event rate does not follow the experience:** about 10 per minute in S35 as well as in S04 and S12 (scores 1214 and 1317). It is set by the threshold.

### Videos and chain

- **S28 (score 1158) versus S35 (score 325), same method:** S28 generates a dense garden illustration and, from 3:30 onward, colored discs with geometric motifs and repeated arrangements of colored objects. S35 generates mostly animals in vegetation, with a stretch of flower pattern between 4:15 and 5:20.
- **The look of S28 depends on the model:** with the five-participant model, DMT gives a grid of colored grains and eyes closed gives a woman in a bedroom; with the one-participant model, the same DMT gives a green backyard.
- **Chain with image prior:** the continuity comes entirely from the prior. With the prior, parts 6 to 20 are almost identical; without the prior, the same vectors give scenes unrelated to each other.

### Split-half test

Frame-by-frame agreement between the trajectories of even and odd events, after removing the mean of each half. Null: shuffling the vectors of one half in time.

| | Five-participant model | One-participant model |
|---|---|---|
| S28, DMT | 0.41 (p = 0.009) | 0.02 (p = 0.40) |
| Group, DMT (mean; p above zero) | 0.11 (p = 0.02; n = 25) | 0.08 (p = 0.03; n = 24) |
| Group, eyes closed | 0.04 (p = 0.27) | 0.00 (p = 0.59) |
| Group, eyes open | 0.00 (p = 0.67) | 0.07 (p = 0.15) |
| DMT × eyes closed, paired | p = 0.36 | p = 0.30 |

- **The S28 case does not survive the model swap**, and 3 of 25 participants pass p < 0.05 on DMT, close to what is expected by chance.
- **In the group, the halves agree slightly more than chance on DMT, in both models**, with no significant difference from eyes closed.
- **No relation to the questionnaires** (ρ between −0.35 and 0.24 depending on the model and the scale, none significant) nor to the number of events.

### Effect curve

- **The effect is in the EEG:** posterior alpha on DMT at 35% of the same participant's eyes-closed level (median), below 1 in 33 of 34 participants. Eyes closed stays flat over time.
- **There is a time course in the group:** in the first 2 min, relative to after 3 min, alpha is more suppressed (p = 0.015), slow waves are stronger (p = 0.0001) and amplitude is larger (p = 0.002). There is no return to rest within the recording.
- **The onset was not recorded:** none of the 35 files has a marker, and in the first 30 s alpha is already at about a quarter of rest.
- **The fast-rise, slow-decay curve does not beat the alternatives:** median R² of 0.24, against 0.30 for the mirrored curve and 0.15 for the straight line (curve × mirrored, p = 0.43).
- **High frequency (a muscle indicator) grows over the recording** (p = 0.02), which confounds "time" with "muscle".
- **The depth of the alpha decrease does not follow the questionnaire total score** (ρ = 0.05).

### Effect direction in the vectors

Per-participant regression: vector = rest + drug level × direction + muscle × direction. About 350 windows per recording, aligned to the detector peaks.

| Question | Result |
|---|---|
| Does the direction repeat across halves of the same participant? | median cosine 0.32 (p = 0.005) |
| Do participants share the direction? | consistency 0.14 (p = 0.001; n = 25) |
| Is DMT − eyes closed (same windows) consistent? | consistency 0.21 (p = 0.001) |
| Effect direction × muscle direction | cosine 0.57 per participant (p = 0.0004); 0.80 between the group means |
| Effect size × questionnaires | ρ between −0.25 and 0.16, none significant |

- **Nearest concepts to the effect direction:** shoulder, elbow, wrist, knee, leg, platypus, seal. Those of the DMT − eyes closed contrast: platypus, anteater, hippopotamus, kangaroo.
- **The direction is that of the spectrum:** less alpha and more fast activity push the vector to the same side. It is the "people and bodies" axis of the first phase, found again by another route.
- **With all windows, without alignment, almost nothing shows up** (consistency 0.05; p = 0.03; halves p = 0.21). The decoder is linear in the waveform, and the mean of out-of-phase windows barely depends on the state.

### Dense decoding and agreement

Source: `reports/dmt_dense_report.json`. Halves by alternating 10 s blocks; null by circular shift.

| | Thresholded events | Dense (all peaks) |
|---|---|---|
| Mean agreement of the halves, DMT | 0.11 | 0.15 (group p = 0.008) |
| Recordings with p < 0.05, DMT | 3 of 25 | 10 of 26 |
| Eyes closed / eyes open | — | 11 of 27 / 10 of 22 |

- **More data per frame makes the trajectory reproducible in more recordings**, but equally without the drug: it is the slow structure of any recording.
- **The colored shapes of S28 persist with eight times more data**: animals in the first minute and a half, saturated patterns from 1:50 to the end. In S04 the order is reversed: patterns in the first minute, then light-colored objects.
- **Frame-by-frame agreement (median):** S28 DMT 0.50; S28 eyes closed 0.18; S04 DMT 0.25; S29 DMT 0.74.

### Patterns, stages and drug level

Sources: `reports/pattern_score.json`, `reports/stage_kinetics.json`. Spearman correlation, within each DMT recording, between the score of each category (text probes) and the measured drug level; 25 participants.

| Category | Real signal | Spectrum only (phase-randomized surrogate) | Real − local surrogate |
|---|---|---|---|
| Geometry | −0.22 (p = 0.02) | −0.16 (p = 0.01) | −0.10 (p = 0.40) |
| Architecture | −0.18 (p = 0.008) | −0.16 (p = 0.01) | −0.12 (p = 0.49) |
| Entities | +0.23 (p = 0.002) | +0.24 (p = 0.01) | +0.06 (p = 0.87) |
| People | +0.20 (p = 0.03) | +0.30 (p = 0.01) | +0.05 (p = 0.62) |
| Objects | −0.03 (p = 0.49) | −0.03 | +0.03 |

By fifth of the recording time (24 participants, deviation from the participant's mean, ×100):

| | 1st (peak) | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Drug level | 1.66 | 1.44 | 1.22 | 1.10 | 1.13 |
| Entities | +3.0 | −0.3 | −0.7 | −1.8 | −0.3 |
| People | +3.3 | −0.6 | −1.1 | −1.8 | +0.1 |
| Animals | +2.6 | +0.6 | −0.7 | −1.6 | −0.8 |
| Geometry | −3.0 | +0.1 | +0.7 | +1.7 | +0.5 |
| Architecture | −4.4 | 0.0 | +1.5 | +2.6 | +0.4 |

- **One axis with two poles:** "beings" (entities, people, animals) at the peak and "structure" (geometry, architecture, natural textures) as the effect declines. Animals follow entities (r = 0.82) and oppose geometry (−0.82).
- **Compatible with the reported stages** for the observable stretch (peak and offset). The initial prediction of architecture at the peak failed; grouping it with geometry was an adjustment made after seeing the data.
- **The spectrum alone reproduces everything**, and the local difference shows nothing. It is the decoder's translation of the state, not a reading of content.
- **Colored pattern per participant:** DMT × eyes closed with no difference (p = 0.40). With the questionnaire total score, ρ = 0.36 (p = 0.07; n = 26); with elementary imagery, ρ = 0.29 (p = 0.15).
- **Reading the videos:** each frame is a sample of a tendency (a region of image space); the concrete exemplar comes from the seed and the generator.

### Calibrated confidence and lens

Sources: `reports/lens_model.json`. Figure: `figures/video_frames_S28_calibrated.jpg`.

- **Effective dimension of the vectors: 4.3.** Three components hold 75% of the variance and ten, 91%. Halves of recordings from different participants have a cosine with mean 0 and standard deviation 0.45 in the original space; in the whitened space (50 components), standard deviation 0.14.
- **Agreement at the same instant × 2 min apart** (original space): S28 0.43 × 0.17; S29 0.59 × −0.07; S04 0.22 × −0.05; S35 0.26 × 0.29. In S35 the agreement comes only from the fixed look of the recording.

| Recording (DMT) | z of the data, 20 s window | z of the prediction | Mean visibility | Frames that are almost only static |
|---|---|---|---|---|
| S28 | 2.1 | 2.3 | 0.71 | 8% |
| S29 | 2.5 | 2.5 | 0.74 | 0% |
| S12 | 1.4 | 1.1 | 0.35 | 30% |
| S04 | 1.2 | 0.9 | 0.31 | 34% |
| S35 | 1.0 | 0.8 | 0.17 | 74% |

Frames with z > 2 by window width (to each side):

| | 5 s | 10 s | 20 s | 40 s |
|---|---|---|---|---|
| S28 | 13% | 27% | 56% | 98% |
| S29 | 19% | 39% | 70% | 100% |
| S35 | 3% | 9% | 11% | 20% |

Inferred lens (smallest window for the target reliability):

| Recording | Consistency between two windows | Lens, target 0.5 (median; p10–p90) | Lens, target 0.65 | Reliability obtained (target 0.5) |
|---|---|---|---|---|
| S29 | 0.019 | 20 s (11–31) | 38 s | 0.52 |
| S28 | 0.010 | 40 s (19–58) | 58 s | 0.59 |
| S04 | 0.004 | 52 s (27–60) | 60 s (cap) | 0.43 |
| S35 | 0.003 | 53 s (28–60) | 60 s (cap) | 0.37 |

- **A single window carries 1% to 2% consistency**; 50 to 100 are needed for the mean to be worth anything. The model's prediction matches the independent halves.
- **Windows below 15 s are almost never justified** (6% of the S28 frames at target 0.5). The stepped lens used them in 47% of the S28 frames.
- **With target 0.65 and a 90 s cap, the S28 lens sits at a median of 70 s (32 to 90)**, with full confidence in the data in practically all frames.
- **Category meter, mean weight over the recording:** S28 geometry 30% and architecture 25%; S35 geometry 7% and architecture 5%. S28 and S29 go from animals, people and entities at the start to architecture and geometry at the end; S35 stays on beings and objects.

### Rest contrast, electrode repair and final cases

Sources: `cases/README.md`, scripts `repair_recording.py`, `dmt_events_video.py … contrast`.

**Contrast against rest** (`contrast`): each window vector has the mean vector of the same subject's eyes-closed recording subtracted. Confidence rises (S04: lens median 27 s, 100% of frames visible) because the DMT − rest difference is a constant component shared by all moments; part of the "beings" content of contrast videos is that constant, which at group level maps to animals and bodies and comes from the spectrum.

**Electrode repair** (`repair_recording.py`): 1 Hz high-pass, then spherical-spline rebuild of the channels that alone drive most rejected windows.

| Recording | Clean before | Clean after | Rebuilt |
|---|---|---|---|
| S27 DMT | 28% | 97% | T8 |
| S05 DMT | 0% | 82% | Fp1, Fp2, P4 |
| S14 DMT | 0% | 70% | Fp2, FC1 |
| S30 DMT | 0% | 97% | FC2 |
| S24 DMT | 1% | 6% | not recoverable (all channels) |

- **Rejections are slow, large-amplitude activity, not muscle**, mostly from a single electrode.
- **Repair changes the category profile.** Applying the repair to subjects with good data (structure / beings share of the contrast vector):

| Treatment | S07 | S12 | S28 |
|---|---|---|---|
| Original | 0.45 / 0.11 | 0.60 / 0.27 | 0.53 / 0.21 |
| High-pass only | 0.57 / 0.12 | 0.60 / 0.23 | 0.45 / 0.36 |
| High-pass + P4 rebuilt | 0.16 / 0.51 | 0.38 / 0.50 | 0.50 / 0.29 |

  One rebuilt posterior electrode can invert the profile. Repaired subjects are therefore excluded from the analysis, and the category balance should be read as fragile.

**Category profile vs questionnaire** (contrast vectors, 30 s kernel; structure = geometry + architecture, beings = entities + people + animals):

| Comparison | All (n = 30) | Unrepaired only (n = 25) |
|---|---|---|
| Structure × reported elementary imagery | ρ = 0.34, p = 0.06 | ρ = 0.38, p = 0.06 |
| Beings × reported complex imagery | ρ = 0.10, p = 0.59 | ρ = 0.08, p = 0.71 |

- **The order of stages is not consistent across individuals**: beings then structure in S28; structure then beings in S12 and S04; a single pole in S07.
- **Lens cap.** With the lens free (up to 90 s) each frame averages minutes and the video dwells on few scenes (S12: median 62 s). The final videos cap the lens at 30 s and show static where that is not enough.

## 14. Overall reading

1. **What the models detect reliably:** **that** the brain under DMT changed, and **when**. In OSF, the effect has the expected time course and placebo is subtracted.
2. **What ATM shows:** the translation of the **rhythm change** (mostly the alpha decrease) into the THINGS vocabulary, that is, people and bodies. The same image comes out of a sober EEG with a DMT spectrum, and different spectral biases generate different categories.
3. **Hints that go beyond rhythm:** in the linear model, DMT generates people and the transplant does not. In the ATM detail pathway, there is a consistent residual of light, translucent surfaces. Neither of the two follows the reported intensity of the imagery.
4. **There is no signal of geometry, mandalas or fractals**, nor any relation to the individual complex imagery scores.
5. **The two DMT datasets diverge in content:** people in Zenodo, textures in OSF. They have different equipment, electrodes and encoders.
6. **The positive control failed:** with sober participants watching clouds in another lab, the model does not decode cloud. Clouds, music and meditation produce the same "people" axis as DMT. **Conclusion of this approach:** the decoders trained on THINGS detect a change in brain state, but do not transfer visual content between labs. There is no evidence of reading the content of DMT visions.
7. **The imagery control closes the argument:** with a sober participant imagining a simple, familiar object, on the same equipment and with a classifier trained on that same participant, the imagined content barely shows up (and what shows up seems to come from the eyes). Scalp EEG captures perception well within the lab, imagery poorly, and the models do not transfer between labs. Decoding DMT visions would require all three at the same time.
8. **The model depends on the exact timing of the image** (40 ms of error is enough to break it), and without that timing very little is left. This explains why, on DMT, it only read the spectrum. The central problem becomes finding out **when** and **in which window** to decode.
9. **Within a lab, "when, how much and what" have a good answer, and the model trained on a few participants works on a new participant.** What does not transfer is between labs and protocols. Since the DMT lab has no visual stimuli for calibration, no model could be validated there, and event-based decoding showed no DMT-specific signal. **The route with a real chance is to record, with the same device as the DMT data, sober participants viewing images one at a time.**
10. **Within each recording, what organizes the vectors is the drug level, and it moves them in the direction of the spectrum.** The effect is reproducible per participant and shared across participants, but it coincides with the muscle direction, lands on the same concepts as the first phase and does not follow the questionnaires. The content left after removing the state (the event-based videos) does not repeat across models. The artifact filter, moreover, selects those who reported the least.
11. **Current best guess:** over a DMT recording, the state moves from "living beings" at the peak to "structure" during the decline, in about 20 of 25 participants, in the order that the reports describe for the observable stretch. The origin is the alpha decrease and recovery translated by the decoder; the phase-randomized surrogate tells the same story. The videos are materializations of that tendency, not what each participant saw.
12. **Controls and what remains.** The pipeline produces colored patterns from any eyes-closed EEG, and the phase-randomized surrogate reproduces the category course. A between-subject tendency remains: the amount of geometric structure in the decoded signal follows the reported intensity of elementary imagery (ρ ≈ 0.35, p ≈ 0.06), with no counterpart for beings and scenes.
