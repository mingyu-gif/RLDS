# UR5e Pick-and-Place RLDS Dataset

A [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) / [RLDS](https://github.com/google-research/rlds)-format dataset builder for UR5e robot arm pick-and-place demonstrations, built for fine-tuning [OpenVLA-OFT](https://github.com/moojink/openvla-oft) Vision-Language-Action models.

Raw demonstrations are recorded as ROS2 bags (RealSense RGB, joint states, gripper I/O) and converted into RLDS-formatted TFRecords compatible with the Open X-Embodiment / OpenVLA training pipeline.

## Overview

- **Robot**: UR5e (6-DoF arm + parallel gripper)
- **Camera**: Intel RealSense (848×480, resized to 224×224 with center-crop)
- **Task**: Pick-and-place, auto-segmented into three stages based on gripper state
  - `pick object` → `place object` → `move to initial position`
- **Action space**: 7-DoF (6 joint positions + 1 gripper state), next-state action shifting applied
- **Language instructions**: Per-stage natural language prompts, embedded with the Universal Sentence Encoder (512-dim)

## Data Format

Each episode follows the standard RLDS schema:

| Field | Shape / Type | Description |
|---|---|---|
| `observation.image` | `(224, 224, 3)`, `uint8` | Center-cropped, resized RGB frame |
| `observation.state` | `(7,)`, `float32` | 6 joint positions + gripper state |
| `action` | `(7,)`, `float32` | Next-step joint + gripper target |
| `language_instruction` | `string` | Stage-specific instruction |
| `language_embedding` | `(512,)`, `float32` | USE embedding of the instruction |
| `discount`, `reward`, `is_first`, `is_last`, `is_terminal` | scalar | Standard RLDS episode flags |

## Repository Structure

```
.
├── example_dataset_dataset_builder.py   # TFDS GeneratorBasedBuilder: parses ROS2 bags → RLDS episodes
├── create_example_data.py               # Script for generating example/dummy data
├── data_tester.py                       # Sanity-check script for verifying dataset output
├── sync_tester.py                       # Timestamp/topic sync verification across ROS2 topics
├── checksums.tsv                        # TFDS download checksums
├── CITATIONS.bib                        # Citation metadata
└── tfds/                                # TFDS build artifacts
```

## Requirements

```bash
pip install tensorflow tensorflow-datasets tensorflow-hub opencv-python numpy rosbags
```

## Usage

### 1. Build the dataset

Point the builder at your recorded ROS2 bag directories (edit the glob path in `_split_generators`), then run:

```bash
cd RLDS
tfds build
```

This parses each rosbag2 episode, extracts synchronized image/joint/gripper streams, auto-labels pick/place/return stages from gripper transitions, and writes out RLDS-formatted TFRecords under `~/tensorflow_datasets/`.

### 2. Verify the output

```bash
python data_tester.py
```

### 3. Use for OpenVLA-OFT fine-tuning

Once built, the dataset can be loaded directly as a registered TFDS dataset (`ur5e_pick_and_place_dataset`) in the OpenVLA-OFT LoRA fine-tuning pipeline.

## Notes

- Frames are only kept once both a valid (non-zero) joint reading and a known gripper state have been observed, to avoid corrupting early frames of an episode.
- Stage transitions are one-directional (`pick → place → return`) and driven by gripper open/close events, not by fixed frame counts.
- The final action of each episode is set to a self-referential (stay) action, and `reward=1.0` is assigned to the terminal step.

## Citation

The RLDS schema and fine-tuning conventions used in this repository follow [OpenVLA](https://github.com/openvla/openvla) and [OpenVLA-OFT](https://github.com/moojink/openvla-oft). If you use this repository, please cite both (see `CITATIONS.bib`):

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}

@article{kim24openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and Xiao, Ted and Balakrishna, Ashwin and Nair, Suraj and Rafailov, Rafael and Foster, Ethan and Sanketi, Pannag and Vuong, Quan and Kollar, Thomas and Burchfiel, Benjamin and Tedrake, Russ and Sadigh, Dorsa and Levine, Sergey and Liang, Percy and Finn, Chelsea},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```
