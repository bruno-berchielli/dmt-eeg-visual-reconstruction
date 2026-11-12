# Cases

Six participants of a public study in which people inhaled DMT while wearing an EEG cap ([Zenodo 3992359](https://zenodo.org/records/3992359)). They are the participants with the strongest experiences, by questionnaire, among those whose recordings needed no electrode repair. Each video shows what their brain signal during DMT resembles, in images, after subtracting their own sober eyes-closed rest.

New to the project? Start with the [README](../../README.md); terms are defined in the [glossary](../glossary.md).

## How to read a frame

- **Picture:** drawn by an image generator from the brain signal of that moment. It turns to static when the signal does not support a picture.
- **Top strip:** time in the session; `lens`, the seconds of EEG averaged for the frame; `data` and `prior`, two confidence scores (how well the local signal holds up, and how well a prediction from the strength of the drug effect holds up).
- **Bottom strip:** the three kinds of image the signal leans toward, out of seven (geometry, architecture, entities, people, animals, nature, objects). The percentages are relative weights, not probabilities.

The kind of image is read from the EEG. The specific picture is the generator's illustration of it, not a recording of what the person saw.

## Videos

Columns:

- **Reported patterns / scenes:** what the participant reported on the questionnaire (0-100): simple imagery such as colors and geometric patterns, and complex imagery such as scenes, beings and objects. Group medians are 85 and 53.
- **Structure / beings in the video:** share of the readout going to geometry and architecture, and to entities, people and animals, averaged over the session.
- **Leading category:** the strongest category in the first, middle and last third of the session.
- **Trusted frames:** share of frames with confidence above 75%.

| Participant | Overall intensity (questionnaire total) | Reported patterns / scenes | Structure / beings in the video | Leading category by third | Trusted frames | Video |
|---|---|---|---|---|---|---|
| S12 | 1317 | 90 / 73 | 0.60 / 0.27 | geometry, architecture, architecture | 70% (23% static) | [download](S12_dmt_minus_rest.mp4?raw=true) |
| S04 | 1214 | 85 / 72 | 0.15 / 0.54 | geometry, objects, animals | 89% | [download](S04_dmt_minus_rest.mp4?raw=true) |
| S07 | 1168 | 83 / 33 | 0.45 / 0.11 | objects, architecture, nature | 89% | [download](S07_dmt_minus_rest.mp4?raw=true) |
| S28 | 1158 | 81 / 59 | 0.53 / 0.21 | animals, geometry, geometry | 100% | [download](S28_dmt_minus_rest.mp4?raw=true) |
| S19 | 1010 | 87 / 48 | 0.32 / 0.23 | objects, architecture, objects | 18% (25% static) | [download](S19_dmt_minus_rest.mp4?raw=true) |
| S31 | 968 | 84 / 57 | 0.25 / 0.44 | people, architecture, objects | 90% | [download](S31_dmt_minus_rest.mp4?raw=true) |

| | |
|---|---|
| S12 | ![S12](S12_frames.jpg) |
| S04 | ![S04](S04_frames.jpg) |
| S07 | ![S07](S07_frames.jpg) |
| S28 | ![S28](S28_frames.jpg) |
| S19 | ![S19](S19_frames.jpg) |
| S31 | ![S31](S31_frames.jpg) |

## What stands out

- **S07** reported strong patterns and few scenes or beings, and the video is almost entirely geometric: only 11% of the readout goes to living things, among the lowest of 30 participants.
- **S12 and S28** reported both patterns and scenes. Both appear in their videos, in opposite order.
- **S04** opens with patterns and moves to objects and beings.

## Scope

The decoder was trained on other people, on different equipment, looking at photos of everyday objects. The videos therefore show how the brain state evolves, expressed in image categories, rather than the specific things a person saw. The sessions have no minute-by-minute report, so the order of scenes cannot be checked against the experience.

To generate one of these videos:

```
cd src && python dmt_events_video.py S07 T500multi DMT inferred contrast step=1 target=0.65 max=30
```
