# Glossary

Terms and abbreviations used across the documentation.

## The experiment

| Term | Meaning |
|---|---|
| EEG | Electroencephalography: electrical activity of the brain recorded by electrodes on the scalp |
| DMT | N,N-dimethyltryptamine, a short-acting psychedelic. In the main dataset it was inhaled; effects last a few minutes |
| Session, recording | One continuous EEG recording of one participant in one condition |
| DMT / EC / EO | The three conditions of the main dataset: under DMT, sober with eyes closed (EC), sober with eyes open (EO). EC is also called "rest" |
| S07, S12, … | Participant codes of the main dataset |
| 5D-ASC | A standard questionnaire on altered states of consciousness, answered after the session. Scores used here: elementary imagery (colors, shapes, patterns), complex imagery (scenes, beings, objects), and the total |
| Alpha rhythm | A brain rhythm around 10 Hz, strong at rest with eyes closed. DMT suppresses it |
| Drug level | How strong the drug effect is at a given moment, measured as alpha suppression relative to the same participant's rest |

## Datasets

| Term | Meaning |
|---|---|
| Zenodo 3992359 | The main dataset: inhaled DMT, 35 participants, 24 electrodes |
| OSF wujgp | Intravenous DMT vs placebo, 12 participants, 6 electrodes |
| PsiConnect | Psilocybin dataset with EEG, used for control experiments |
| THINGS-EEG2 | Sober volunteers viewing about 16,000 photos of everyday objects, shown at 5 per second. The training data for the decoders |
| Imagery dataset | 22 sober volunteers seeing and then imagining 10 pictures, one at a time (Figshare 30227503) |
| Lab | A recording site with its own equipment and protocol. "Cross-lab" means training on one dataset and testing on another |

## Models

| Term | Meaning |
|---|---|
| Decoder, encoder | A model that maps an EEG window to an image embedding |
| Embedding, vector | A list of 1024 numbers describing an image, in the space of the CLIP model. Similar images have similar vectors |
| CLIP ViT-H/14 | The image model that defines that space |
| ATM, ATM-S | The EEG decoder architecture of Li et al. (2024), reimplemented here |
| Ridge | A linear decoder. The videos use a ridge trained on five THINGS-EEG2 participants |
| SDXL-Turbo, IP-Adapter | The image generator and the adapter that lets it draw from an embedding instead of a text prompt |
| Top-1, top-5 | Decoding accuracy: the right image ranked first, or among the first five, out of 200 candidates |

## Analysis

| Term | Meaning |
|---|---|
| Window | One second of EEG, the unit the decoder reads |
| Onset | The instant an image appears. The decoder only works when its window starts within about 40 ms of it |
| Template, detector | A matched filter that finds moments in a recording resembling the brain's response to an image onset. Used to align windows |
| Event | A detector peak above a threshold. Early analyses used only events; later ones use every peak ("dense" decoding) |
| Phase-randomized surrogate | A scrambled copy of a recording with the same rhythms (spectrum) and no timing structure. Shows what the pipeline outputs for the spectrum alone |
| Spectrum | How much of each rhythm (frequency) a signal contains |
| Contrast | A difference between conditions for the same participant, usually DMT minus rest |
| Lens | The width of the time window averaged for one video frame |
| Split-half | Building the same quantity from two independent halves of the data and comparing them. Agreement means the result is in the data, not in the sampling |
| Reliability | Split-half agreement on a 0 to 1 scale |
| Data / prior confidence | Per frame: how well the local data holds up, and how well the prediction from the drug level holds up |
| Static | Noise shown instead of a picture when neither confidence holds |
| Structure / beings | The two poles of the category readout: geometry and architecture; entities, people and animals |
| Text probes | Short text descriptions (for example "a mandala", "a dog") used to name the categories a vector leans to |
| Repair | Rebuilding a faulty electrode from its neighbours so that a recording is not discarded |
| p | Probability of a result at least this strong arising by chance. Below 0.05 is the usual threshold |
| ρ (rho) | Spearman rank correlation, from −1 to 1 |
