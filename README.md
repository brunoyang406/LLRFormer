# LLRFormer

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-orange.svg)](https://pytorch.org/)

**LLRFormer** (Lower Limb Radiographs Automatic Landmark Detection and Alignment Measurements via Transformer) is a Transformer-based keypoint detection model for full-length lower limb radiographs. The model combines HRNet backbone with cross-self attention mechanism, specifically designed for medical image keypoint detection tasks.

## Features

- **Medical Image Specialized**: Optimized Landmark Detection model for full-length lower limb radiographs
- **High Precision Detection**: Supports accurate detection of 36 keypoints (hip, femur, knee, tibia, ankle)
- **Cross-Self Attention**: 6-layer cross-self attention Transformer for enhanced feature interaction

## Architecture

LLRFormer adopts a two-stage architecture:

1. **Backbone**: HRNet-W32 for multi-scale feature extraction
2. **Transformer Module**: 6-layer cross-self attention mechanism for keypoint feature processing

### Architecture Specifications

- **Input Size**: 384×1152 (width×height)
- **Number of Keypoints**: 36
- **Feature Dimension**: 192
- **Attention Heads**: 8
- **Patch Size**: 9×3

## Directory Structure

```
LLRFormer/
├── configs/              # Configuration files
│   └── llrformer.yaml   # Main configuration file
├── data/                 # Dataset directory (prepare yourself)
│   ├── train/           # Training set
│   ├── val/             # Validation set
│   └── test/            # Test set
├── dataset/             # Dataset loaders
│   ├── dataloader.py    # Data loader
│   └── mydataset.py     # Dataset class
├── lib/                  # Core library
│   ├── config/          # Configuration management
│   ├── core/            # Core functions (training, evaluation, inference)
│   ├── models/          # Model definitions
│   │   ├── llrformer.py           # LLRFormer main model
│   │   ├── hr_base.py             # HRNet backbone
│   │   └── transformer_backbone.py # Transformer module
│   └── utils/           # Utility functions
├── pretrained/          # Pretrained weights
│   └── imagenet/       # ImageNet pretrained weights
├── tools/               # Training and testing scripts
│   ├── train.py        # Training script
│   └── test.py         # Testing script
├── experiments/         # Experimental results
│   ├── log/            # Training logs
│   └── output/         # Model outputs
└── visualization/       # Visualization tools
```

## Quick Start

### Requirements

- Python >= 3.6
- PyTorch >= 1.7.0
- CUDA (recommended for GPU acceleration)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/LLRFormer.git
cd LLRFormer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download pretrained weights**

Download HRNet-W32 ImageNet pretrained weights:
```bash
mkdir -p pretrained/imagenet
cd pretrained/imagenet
# Download from official repository: https://github.com/HRNet/HRNet-Image-Classification
# Or use wget:
wget https://github.com/HRNet/HRNet-Image-Classification/releases/download/v1.0/hrnet_w32-36af842e.pth
cd ../..
```

### Data Preparation

We use the full-length lower-limb radiograph dataset (LLRFormer Keypoint Dataset), which is publicly available. The dataset includes training, validation, test, and external test sets, along with corresponding JSON annotation files. It can be accessed at Hugging Face:
https://huggingface.co/datasets/YANG1568279/full_length_lower_limb_radiographs_with_metallic_implants

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── images/          # Training images
│   └── annotations/     # Training annotations (JSON format)
├── val/
│   ├── images/          # Validation images
│   └── annotations/     # Validation annotations
└── test/
    ├── images/          # Test images
    └── annotations/     # Test annotations
```

**Annotation Format**: Use xanylabeling JSON format. Each image should have a corresponding JSON annotation file with the same base name. Annotation files should contain 36 keypoints, which can be `point` or `circle` type.

For detailed data preparation instructions, please refer to [data/README.md](data/README.md).

## Usage

### Training

```bash
python tools/train.py --cfg configs/llrformer.yaml --gpus 0
```

**Training Arguments**:
- `--cfg`: Path to configuration file
- `--gpus`: GPU IDs to use (e.g., `0` or `0,1`)
- `--dataDir`: Data root directory (optional, defaults to path in config file)
- `--modelDir`: Model output directory (optional)
- `--logDir`: Log output directory (optional)

### Testing/Evaluation

```bash
python tools/test.py --cfg configs/llrformer.yaml --gpus 0
```

**Testing Arguments**:
- `--cfg`: Path to configuration file
- `--gpus`: GPU IDs to use
- Model weight path is specified in `TEST.MODEL_FILE` in the config file

### Configuration

Main configuration parameters are in `configs/llrformer.yaml`:

**Dataset Configuration**:
```yaml
DATASET:
  DATASET: 'MyKeypointDataset'
  ROOT: 'data'
  TRAIN_SET: 'train'
  TEST_SET: 'val'
  IMAGE_SIZE: [384, 1152]
```

**Model Configuration**:
```yaml
MODEL:
  NAME: llrformer
  NUM_JOINTS: 36
  USE_CROSS_SELF_ATTENTION: true
  CROSS_SELF_ATTENTION_LAYERS: 6
  TRANSFORMER_DEPTH: 6
  TRANSFORMER_HEADS: 8
  DIM: 192
```

**Training Configuration**:
```yaml
TRAIN:
  BATCH_SIZE_PER_GPU: 8
  END_EPOCH: 250
  OPTIMIZER: adamw
  LR: 0.001
  LR_SCHEDULER: 'ReduceLROnPlateau'
  JOINT_WEIGHTS: true
  FEMUR_TIBIA_WEIGHT: 2.0
```

For more configuration details, please refer to [configs/README.md](configs/README.md).

## Evaluation Metrics

The model is evaluated using the following metrics:

- **PCK@0.02**: Percentage of Correct Keypoints with threshold 0.02
- **AUC**: Area Under Curve


## Dataset Statistics

- **Training set**: 709 images
- **Validation set**: 102 images
- **Test set**: 200 images
- **Number of keypoints**: 36

### Visualization Tools

The `visualization/` directory contains three visualization scripts:

- **`attention_vis_keypoint_attention_layers.py`**: Visualizes cross-keypoint attention matrices across all 6 transformer layers. Output: `attention_vis/cross_keypoint_attention_layers.png`

- **`attention_visualize.py`**: Comprehensive attention visualization including cross-keypoint attention, keypoint-to-image attention maps, and symmetry analysis. Output: `attention_vis/cross_keypoint_attention_layers.jpg` and keypoint-to-image attention maps.

- **`plot_keypoint_token_similarity.py`**: Visualizes keypoint token embedding similarity as a heatmap matrix. Output: `attention_vis/keypoint_token_similarity.png`

**Usage:**
```bash
cd visualization
python <script_name>.py
```

## License

This project is licensed under the [LICENSE](LICENSE) license.

## Acknowledgments

- [HRNet](https://github.com/HRNet/HRNet-Image-Classification): For the backbone network
- [xanylabeling](https://github.com/CVHub520/xanylabeling): For data annotation

## Contact

For questions or suggestions, please submit an Issue or Pull Request.

---

**Note**: This project is for medical image analysis and is for research purposes only. Before clinical application, ensure the model has been thoroughly validated and tested.
