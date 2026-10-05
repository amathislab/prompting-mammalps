# SALMA

SALMA detects, tracks and describes animals in MammAlps-S2 videos. A VideoMAE ViT encoder feeds a DETR-style decoder with animal queries. For each video it predicts boxes, track identities, species, activity, actions, deer age and sex, and weather. Its per-video JSON predictions are the input of the agent-based baseline of Prompting-MammAlps.

This folder contains the code to train SALMA and to run inference.

## Installation

From the repository root:

```bash
uv pip install -e ".[salma]"
```

Install a `torch`/`torchvision` build that matches your CUDA version. All commands below are run from `models/salma/`.

## Data

Download the dataset into `./data` as described in the [root README](../../README.md). The scripts read:

```
data/
├── metadata/
│   ├── train.csv              # video_file_path,annotation_file_path,duration,fps,sampling_weight
│   ├── test.csv
│   └── label_mapping.json     # species, activities, actions, deer_ages, deer_adult_sexes, weather
├── videos/{train,test}/S*/C*/*.mp4
└── annotations/{train,test}/S*/C*/*.json
```

There is no validation split, so training always runs with `--no_val`.

## Weights

```bash
mkdir -p pretrained_models
# InternVideo ViT-L encoder (only needed for training)
wget -P pretrained_models https://huggingface.co/OpenGVLab/InternVideo1.0/resolve/main/internvideomae_classification/vit_l_hybrid_pt_800e.pth
```

## Training

```bash
bash scripts/train.sh
```

Training followed a two-stage curriculum. Each stage runs for 500 epochs on 8 GPUs:

1. **Stage 1** starts from the InternVideo encoder and learns localization and tracking only. The box, GIoU, objectness and contrastive losses are on, and the classification heads are frozen.
2. **Stage 2** starts from the stage 1 checkpoint and trains all heads (species, activity, actions, deer age and sex, weather).

Key options of `train_salma.py`:

| Flag | Meaning |
|---|---|
| `--encoder_path` / `--ckpt_path` | Initialize the encoder only / the full model |
| `--loss_<name>` | Weight of each loss term (0 disables it) |
| `--freeze_object_heads` | Freeze the classification heads |
| `--rand_freq --min_rts --max_rts` | Random temporal stride sampling during training |
| `--jitter` | Temporal jittering |
| `--balancing_sampling` | Sample videos by `sampling_weight` |
| `--use_lora` | LoRA fine-tuning of the encoder |

The encoder is frozen for the first `--warmup_epochs`. Learning rates are scaled by `batch_size × num_gpus / 256`.

## Inference

```bash
bash scripts/inference.sh
```

Each test video is processed as two overlapping square crops (left and right), and the two sets of predictions are merged:

```
output/salma_large_predictions/
├── left/<video_id>.json
├── right/<video_id>.json
└── multi_view/<video_id>.json   # merged predictions
```

`multi_view` files use the same schema as `data/annotations/`. Each file has an `info` block with video-level attributes, and a `frames` list of detections. Every detection has a `track_id`, a `bbox`, a `conf` and per-animal `attributes`. These files can be passed directly to `models/llm/evaluate_llm_functions.py` and `tutorials/visualize_predictions.ipynb`.

## Acknowledgements

This code builds on [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [InternVideo](https://github.com/OpenGVLab/InternVideo), [TIM](https://github.com/JacobChalk/TIM), [DETR](https://github.com/facebookresearch/detr) and [timm](https://github.com/huggingface/pytorch-image-models). See [NOTICE](NOTICE) for details.
