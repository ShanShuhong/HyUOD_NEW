# HyUOD: Hybrid Underwater Object Detection

## Introduction

HyUOD is a hybrid framework for underwater object detection (UOD) that combines
multi-scale feature fusion with underwater image enhancement to improve detection
performance in challenging underwater environments.

Key features:
- **Hybrid architecture**: Fuses raw and enhanced image features for complementary information
- **Multi-scale detection**: Feature Pyramid Network (FPN) neck for detecting objects at different scales
- **Underwater-aware training**: Augmentations and loss functions tailored for underwater scenes
- **DUO dataset support**: Built-in dataset loader for the Detecting Underwater Objects (DUO) benchmark

## Installation

```bash
# Create conda environment
conda create -n hyuod python=3.8 -y
conda activate hyuod

# Install PyTorch (adjust cuda version as needed)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install dependencies
pip install -r requirements.txt

# Install HyUOD
pip install -e .
```

## Dataset Preparation

Download the [DUO dataset](https://github.com/chongweiliu/DUO) and organize it as follows:

```
data/
└── DUO/
    ├── images/
    │   ├── train/
    │   └── test/
    └── annotations/
        ├── train.json
        └── test.json
```

## Training

```bash
# Single GPU
python tools/train.py configs/hyuod_r50_fpn_duo.py

# Multi-GPU distributed training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 tools/train.py configs/hyuod_r50_fpn_duo.py \
    --launcher pytorch
```

## Testing

```bash
python tools/test.py configs/hyuod_r50_fpn_duo.py \
    work_dirs/hyuod_r50_fpn_duo/latest.pth \
    --eval bbox
```

## Results on DUO Dataset

| Method | mAP | holothurian | echinus | scallop | starfish |
|--------|-----|-------------|---------|---------|---------|
| HyUOD  | -   | -           | -       | -       | -       |

*(Results will be updated after training)*

## Acknowledgement

- [DUO dataset](https://github.com/chongweiliu/DUO)
- [GCC-Net](https://github.com/Ixiaohuihuihui/GCC-Net)
- [MMDetection](https://github.com/open-mmlab/mmdetection)

