# References

Legend:
- **consulted**: the page, the README or the text was opened during the experiment;
- **cited**: background reference, not reread in this round.

## Data

| Source | Status | Use |
|---|---|---|
| Pallavicini C. et al. (2021). *Neural and subjective effects of inhaled N,N-dimethyltryptamine in natural settings.* J. Psychopharmacology. Data: [Zenodo 3992359](https://zenodo.org/records/3992359) | consulted (record and files) | main DMT dataset |
| Alamia A., Timmermann C., Nutt D., VanRullen R., Carhart-Harris R. (2020). *DMT alters cortical travelling waves.* eLife 9:e59784. Data: [OSF wujgp](https://osf.io/wujgp/) | consulted (full text via Europe PMC: [PMC7577737](https://europepmc.org/article/PMC/PMC7577737)) | intravenous DMT × placebo, 5 midline electrodes |
| Timmermann C. et al. (2019). *Neural correlates of the DMT experience assessed with multivariate EEG.* Scientific Reports | cited | source study of the OSF data |
| Novelli L. et al. (2026). *PsiConnect.* Scientific Data. Stoliker D. et al. (2026). *Psychedelics align brain activity with context.* Nature. Data: [OpenNeuro ds006110](https://openneuro.org/datasets/ds006110) | consulted (README, metadata, phenotypes) | psilocybin 19 mg × baseline, 64-channel EEG |
| Gifford A.T. et al. (2022). *A large and rich EEG dataset for modeling human visual object recognition* (THINGS-EEG2). NeuroImage | cited | encoder training |
| Hebart M.N. et al. (2019). *THINGS: A database of 1,854 object concepts…* PLoS ONE | cited | concept vocabulary |
| [HF LidongYang/EEG_Image_decode](https://huggingface.co/datasets/LidongYang/EEG_Image_decode) | consulted (README and listing) | THINGS-EEG2 preprocessed at 250 Hz, CLIP ViT-H/14 features, images |
| *An EEG Dataset for Visual Imagery-Based Brain–Computer Interface.* Scientific Data (2025). [Paper](https://www.nature.com/articles/s41597-025-06512-5), data: [Figshare 30227503](https://doi.org/10.6084/m9.figshare.30227503) | consulted (full text via Europe PMC, README, code) | imagery control: seeing × imagining 10 images |
| Wilson H. et al. (2023). *EEG-based BCI Dataset of Semantic Concepts for Imagination and Perception Tasks.* Scientific Data. [OpenNeuro ds004306](https://openneuro.org/datasets/ds004306) | consulted (search result) | alternative: 3 concepts, 124 channels |
| *A Human EEG Dataset for Multisensory Perception and Mental Imagery.* Scientific Data (2025). [OpenNeuro ds005815](https://openneuro.org/datasets/ds005815) | consulted (search result) | alternative: multisensory imagery |
| [OSF uxqcv](https://osf.io/uxqcv/), "Ayahuasca Study Data" | consulted | questionnaires only (xlsx + R); no EEG |

## EEG → image/video/text models

| Source | Status | Conclusion |
|---|---|---|
| Li D., Wei C., Li S., Zou J., Liu Q. (2024). *Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion* (ATM). NeurIPS. [arXiv 2403.07721](https://arxiv.org/abs/2403.07721), [repo](https://github.com/ncclab-sustech/EEG_Image_decode) | consulted (README, preprocessing code, model and pipeline) | architecture reimplemented; the encoder weights are not public |
| [EEGMirror](https://github.com/XuanhaoLiu/EEGMirror) | consulted (README) | no published weights; training requires SEED-DV/SEED/FACED |
| Liu X. et al. (2024). *EEG2Video.* NeurIPS. [repo](https://github.com/Skeli9989/EEG2Video) | consulted (search result) | no weights found; SEED-DV requires a license |
| *MindCine: Multimodal EEG-to-Video Reconstruction with Large-Scale Pretrained Models.* [arXiv 2601.18192](https://arxiv.org/abs/2601.18192) | consulted (search result) | successor to EEG2Video; weights not confirmed |
| Perceptogram (2024), linear EEG→CLIP reconstruction | cited | inspired the linear (ridge) decoder |
| Bai Y. et al. (2023). *DreamDiffusion.* | cited | discarded: 128-channel dataset with a block-design confound |
| Li R. et al. (2021). *The perils and pitfalls of block design for EEG classification experiments.* IEEE TPAMI | cited | reason for discarding DreamDiffusion |
| Duan Y. et al. (2023). *DeWave.* NeurIPS. Jo H. et al. (2024). *Are EEG-to-Text Models Working?* arXiv | cited | dedicated EEG→text discarded: similar performance with noise as input |
| [NeuralBench, EEG video decoding](https://facebookresearch.github.io/neuroai/neuralbench/tasks/eeg/video.html) | consulted (search result) | overview of EEG→video tasks |

## Image generation

| Source | Status | Use |
|---|---|---|
| Sauer A. et al. (2023). *Adversarial Diffusion Distillation* (SDXL-Turbo). [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) | cited | generator, 4 steps, 512×512 |
| Ye H. et al. (2023). *IP-Adapter.* [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter), `ip-adapter_sdxl_vit-h` | cited; weights inspected | conditioning on the CLIP embedding |
| OpenCLIP ViT-H/14 LAION-2B ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)) | cited | shared image/text space, text probes |

## Methods and architecture

| Source | Status | Use |
|---|---|---|
| Liu Y. et al. (2024). *iTransformer.* ICLR | cited | cross-channel attention block of ATM |
| Schirrmeister R. et al. (2017). *Deep learning with convolutional neural networks for EEG decoding* (ShallowConvNet) | cited | convolutional block of ATM |
| Li Y. et al. (2016). *Revisiting Batch Normalization for Practical Domain Adaptation* (AdaBN) | cited | label-free recalibration to another domain |
| Dittrich A. (1998); Studerus E. et al. (2010). 5D-ASC / 11D-ASC | cited | subjective scales of altered states |

## Imagery, temporal sweep and spontaneous visions

| Source | Status | Note |
|---|---|---|
| Dijkstra N. et al. (2018). *Differential temporal dynamics during visual imagery and perception.* [eLife](https://elifesciences.org/articles/33904) | consulted (excerpts of the full text via Europe PMC) | imagery shares with perception the representations at ~160 ms and from 300 ms onward, not the early ones; it is stable over time |
| Shatek S., Grootswagers T., Robinson A., Carlson T. (2019). *Decoding images in the mind's eye.* [bioRxiv](https://www.biorxiv.org/content/10.1101/637603v1.full) | consulted | with EEG, imagery not decodable and no generalization from perception; attributed to temporal misalignment |
| Liu Y. et al. (2021). *Temporally delayed linear modelling (TDLM) measures replay in both animals and humans.* [eLife](https://elifesciences.org/articles/66917) | consulted (search result) | precedent for the sweep: train with a marked stimulus and sweep the rest period |
| Kern S. et al. *Challenges in replay detection by TDLM in post-encoding resting state.* [eLife](https://elifesciences.org/articles/108023) | consulted (search result) | low sensitivity; requires very frequent events; resting alpha interferes |
| Trial [NCT07693257](https://clinicaltrials.gov/study/NCT07693257), *Neural Effects of Intravenous DMT* | consulted (record; decoding plan seen only in the search summary) | prolonged intravenous DMT with fMRI, 20 participants, not yet recruiting; validation by choice among distractors |
| Bellec Y. (2025). *Dream2Image.* [arXiv 2510.06252](https://arxiv.org/abs/2510.06252) | consulted (search result) | 38 participants, 129 dreams, 17-channel EEG with reports; access requires registration |
| Horikawa T. et al. (2013). *Neural decoding of visual imagery during sleep.* Science | cited | dream decoding with fMRI, at the category level |

## Side readings

| Source | Status | Note |
|---|---|---|
| Takehi, Teoh, Zhang, Khattab. *Sparks of Weight-Space Programming.* [blog](https://rikiyat.github.io/blog/weight-space-programming/) | consulted | motivated the gradient-free reading and editing of the ATM weights |

## Kinetics and state of the EEG under DMT (cited from memory in the discussion, not reread)

- Timmermann et al. (2019), *Neural correlates of the DMT experience assessed with multivariate EEG*, Scientific Reports. Intravenous DMT with EEG, intensity reported minute by minute and blood concentration; the alpha drop tracks both curves.
- Alamia, Timmermann et al. (2020), *DMT alters cortical travelling waves*, eLife. With eyes closed under DMT, the waves start to travel from back to front, as in visual stimulation. Candidate marker of "when" (not yet tested here).
- Strassman et al. (1994), dose-response studies of intravenous DMT: onset in seconds, peak at 2 to 3 minutes, end at 20 to 30 minutes.

## fMRI decoding and psychedelic fMRI (surveyed, not used)

| Source | Status | Note |
|---|---|---|
| Beliy R. et al. (2025). *Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer.* ICLR 2026. [arXiv 2510.25976](https://arxiv.org/abs/2510.25976) | consulted (abstract, search results) | functional clusters shared across subjects; authors report 1 h of data from a new subject matching 40 h baselines |
| *ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding.* NeurIPS 2025. [arXiv 2510.27128](https://arxiv.org/abs/2510.27128), [repo](https://github.com/xmed-lab/ZEBRA) | consulted (search results) | no data from the new subject required |
| *NSD-Imagery: A benchmark dataset for extending fMRI vision decoding methods to mental imagery.* CVPR 2025. [arXiv 2506.06898](https://arxiv.org/abs/2506.06898) | consulted (search results) | simple linear decoders transfer from perception to imagery better than complex ones |
| Timmermann C. et al. (2023). *Human brain effects of DMT assessed via EEG-fMRI.* [PNAS](https://www.pnas.org/doi/10.1073/pnas.2218949120). Data: [timmer500/DMT_Imaging](https://github.com/timmer500/DMT_Imaging) | files inspected | fMRI parcellated to 112 regions × 840 volumes per session; EEG as band-power regressors; not usable for image reconstruction |
| Carhart-Harris R. et al. (2016), LSD. Data: OpenNeuro ds003059 | consulted (search results) | voxel-level, eyes-closed rest, LSD vs placebo |
| [Reconstructing the visual hallucinations experienced on LSD](https://www.beckleyfoundation.org/reconstructing-the-visual-hallucinations-experienced-on-lsd/) (Beckley Foundation) | consulted | announced project; no published result found |
