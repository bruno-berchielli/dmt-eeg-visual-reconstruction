# Original brief

Text that originated the experiment (2026-11-11). The brief as received ended in the middle of section 3, item A; the rest was never recorded.

## Objective

Investigate whether machine learning models already trained to reconstruct images or videos from EEG can produce visual representations minimally related to the subjective experiences of people under the effect of DMT.

The main task: take real EEGs from DMT experiences, feed them to existing EEG→image or EEG→video models and observe what comes out. No new architecture, no paper and no requirement of rigorous statistical validity: experiment with existing models, check technical feasibility and analyze the results visually.

## Hypotheses

- **Main:** closed-eye visions (CEVs) under DMT produce patterns of brain activity that share characteristics with the activity of a sober person viewing images or videos. If these signals are given to models trained to reconstruct real stimuli, the generated images may have some correspondence with the subjective content of the experience.
- **Secondary:** participants who report geometric, complex or intense visions may produce EEGs that result in images with corresponding characteristics.
- **Also part of the experiment:** the models may simply fail, generate random images or reflect biases learned in training.

## Indicated data

- **Main:** [Zenodo 3992359](https://zenodo.org/records/3992359), *Neural and Subjective Effects of Inhaled DMT in Natural Settings* (2020). About 35 participants, 24-channel EEG in `.bdf`, DMT and control recordings with eyes open and eyes closed, and `scales_results.csv` with variables such as `5D-ElementaryImagery`, `5D-ComplexImagery` and `5D-AudioVisualSyn`.
  - Visual-intensity questionnaires are not equivalent to narrative reports of what the person saw. Look for free-text reports and do not invent descriptions.
  - As an alternative, look for datasets of DMT or other psychedelics with EEG associated with detailed individual reports.
- **Secondary:** [OSF wujgp](https://osf.io/wujgp/), *DMT Alters Cortical Travelling Waves*. Check whether the data are in a reusable format.

## Candidate models

- A. [EEGMirror](https://github.com/XuanhaoLiu/EEGMirror), priority candidate for video (the brief was cut off here).
