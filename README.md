# SMART - Semi-supervised Medical Adaptive vessel Representation Toolkit

[![Hugging Face Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-blue)](https://huggingface.co/papers/2603.00881)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/ly17/TC-SemiSAM-checkpoints)
[![arXiv](https://img.shields.io/badge/arXiv-2603.00881-b31b1b.svg)](https://arxiv.org/abs/2603.00881)

A semi-supervised video vessel segmentation framework built on SAM3 (Segment Anything Model 3), featuring Mean Teacher architecture and text prompt support for efficient coronary angiography vessel segmentation.

## Project Structure

```
SMART/
├── data/                           # Data module
│   ├── __init__.py
│   ├── video_dataset.py            # Video dataset classes
│   └── two_stream_video_sampler.py # Two-stream batch sampler
├── models/                         # Model module
│   ├── __init__.py
│   ├── sam3_video_predictor.py     # SAM3 video predictor
│   └── mean_teacher_sam3.py        # Mean Teacher architecture
├── losses/                         # Loss function module
│   ├── __init__.py
│   ├── supervised_loss.py          # Supervised loss
│   ├── consistency_loss.py         # Consistency loss
│   ├── flow_consistency_loss.py    # Optical flow temporal loss
│   └── confidence_aware_loss.py    # Confidence-aware loss
├── train/                          # Training module
│   ├── __init__.py
│   ├── train_semi_sam3_video.py    # Main training script
│   ├── configs/
│   │   └── semi_sam3_video.yaml    # Training configuration
│   └── utils/
│       ├── __init__.py
│       ├── ema.py                  # EMA updates
│       └── ramps.py                # Weight scheduling
├── scripts/                        # Scripts directory
│   ├── debug/                      # Debug scripts
│   ├── evaluate/                   # Evaluation scripts
│   ├── predict/                    # Prediction scripts
│   ├── visualize/                  # Visualization scripts
│   ├── test/                       # Test scripts
│   ├── utils/                      # Utility scripts
│   └── run_train.sh                # Training startup script
├── submodules/                     # External submodules
│   ├── consema/                    # Conformal Prediction (MICCAI 2025)
│   ├── denver/                     # DeNVeR (CVPR 2025)
│   └── SemiSAM/                    # Semi-supervised SAM methodology
└── README.md                       # This file
```

## Core Components

### 1. Data Loading

- **VideoVesselDataset**: Base video vessel segmentation dataset
- **LabeledVideoDataset**: Labeled video dataset (36 videos, 1220 frames)
- **UnlabeledVideoDataset**: Unlabeled video dataset (96 videos, 5427 frames)
- **TwoStreamVideoBatchSampler**: Two-stream batch sampler for semi-supervised learning

### 2. Model Architecture

- **SAM3VideoPredictor**: SAM3 video predictor with batch frame inference and text prompt caching
- **MeanTeacherSAM3**: Mean Teacher architecture with student model and EMA teacher model
- **ConfidenceAwareMeanTeacher**: Confidence-aware Mean Teacher variant

### 3. Loss Functions

- **SupervisedLoss**: BCE + Dice supervised loss for labeled data
- **ConsistencyLoss**: MSE/KL divergence consistency loss between student and teacher
- **FlowConsistencyLoss**: Optical flow temporal consistency loss
- **ConfidenceAwareLoss**: Uncertainty-weighted consistency loss

## Quick Start

### 1. Environment Setup

Ensure the following dependencies are installed:
- PyTorch >= 1.10
- torchvision
- opencv-python
- PyYAML
- tqdm
- tensorboard

### 2. Configuration

Edit `train/configs/semi_sam3_video.yaml` to modify:
- Data paths
- Training parameters
- Loss function weights

### 3. Start Training

```bash
# Option 1: Using startup script
bash scripts/run_train.sh

# Option 2: Direct execution
python train/train_semi_sam3_video.py --config train/configs/semi_sam3_video.yaml
```

### 4. Monitor Training

```bash
# View TensorBoard
tensorboard --logdir logs/
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 4 | Total batch size |
| labeled_batch_size | 1 | Labeled batch size |
| ema_decay | 0.99 | EMA decay coefficient |
| consistency_weight | 0.1 | Consistency loss weight |
| consistency_rampup | 40 | Consistency loss warmup length |
| lr_transformer | 8e-6 | Transformer learning rate |
| lr_vision_backbone | 2.5e-6 | Vision backbone learning rate |

## Training Pipeline

```
for each iteration:
    1. Sample batch (labeled + unlabeled)
    2. Student model forward pass
    3. Teacher model forward pass (with noisy inputs)
    4. Compute supervised loss (labeled data only)
    5. Compute consistency loss (all data)
    6. Total loss = L_sup + λ_cons * L_cons
    7. Backpropagate to update student
    8. EMA update teacher model
```

## Text Prompt

```python
TEXT_PROMPT = "Please segment the blood vessels"
```

## Input Specifications

- Dataset resolution: 512×512
- SAM3 input resolution: 1008×1008
- Normalization range: [-1, 1] (mean=0.5, std=0.5)

## Changelog

| Date | Changes |
|------|---------|
| 2025-03-01 | Initial SMART repository created |
| 2025-12-12 | Complete training framework implemented |
