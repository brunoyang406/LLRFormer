# LLRFormer Configuration Files

This directory contains configuration files for the LLRFormer model.

## Configuration File

### `llrformer.yaml`

Main configuration file for LLRFormer (Lower Limb Radiographs Automatic Landmark Detection and Alignment Measurements via Transformer), a keypoint detection model using 6-layer Cross-Self Attention mechanism.

#### Model Architecture
- **Backbone**: HRNet-W32
- **Transformer**: 6-layer Cross-Self Attention
- **Keypoints**: 36 keypoints
- **Input Size**: 384×1152
- **Output Heatmap Size**: 96×288

#### Key Features
- Cross-Self Attention mechanism for better feature interaction
- Data augmentation strategies
- Adaptive learning rate scheduling
- Per-keypoint weight loss for imbalanced regions

## Usage

### Training

```bash
python tools/train.py --cfg configs/llrformer.yaml --gpus 0
```

### Testing/Evaluation

```bash
python tools/test.py --cfg configs/llrformer.yaml --gpus 0
```

## Configuration Parameters

### Dataset Configuration
- `DATASET`: Dataset name (`MyKeypointDataset`)
- `ROOT`: Root directory of dataset (`data`)
- `IMAGE_SIZE`: Input image size `[width, height]` = `[384, 1152]`
- `HEATMAP_SIZE`: Output heatmap size `[width, height]` = `[96, 288]` (1/4 of image size)
- `FLIP`: Disabled
- `SCALE_FACTOR`: Disabled
- `ROT_FACTOR`: Disabled

### Model Configuration
- `NAME`: Model architecture name (`llrformer`)
- `NUM_JOINTS`: Number of keypoints (`36`)
- `USE_CROSS_SELF_ATTENTION`: Enable Cross-Self Attention (`true`)
- `CROSS_SELF_ATTENTION_LAYERS`: Number of Cross-Self Attention layers (`6`)
- `TRANSFORMER_DEPTH`: Transformer depth (`6`)
- `TRANSFORMER_HEADS`: Number of attention heads (`8`)
- `DIM`: Feature dimension (`192`)
- `PATCH_SIZE`: Patch size for tokenization `[height, width]` = `[9, 3]`

### Training Configuration
- `BATCH_SIZE_PER_GPU`: Batch size per GPU (`1`)
- `END_EPOCH`: Total training epochs (`250`)
- `OPTIMIZER`: Optimizer type (`adamw`)
- `LR`: Initial learning rate (`0.001`)
- `LR_SCHEDULER`: Learning rate scheduler (`ReduceLROnPlateau`)
- `JOINT_WEIGHTS`: Enable per-keypoint weight loss (`true`)
- `FEMUR_TIBIA_WEIGHT`: Weight multiplier for Femur and Tibia regions (`2.0`)

### Output Paths
- **Model weights**: `experiments/output/MyKeypointDataset/llrformer/llrformer/model_best.pth`
- **Training logs**: `experiments/log/MyKeypointDataset/llrformer/`
- **Checkpoints**: `experiments/output/MyKeypointDataset/llrformer/llrformer/checkpoint.pth`

## Notes

1. **Image Size**: The model expects input images of size 384×1152 (width×height)
2. **Heatmap Size**: Output heatmaps are 96×288, which is 1/4 of the input size
3. **Patch Size**: Patches are 9×3 (height×width), matching the feature map dimensions
4. **Data Augmentation**: Flip, scale, and rotation augmentations are disabled
5. **Region Weights**: Femur and Tibia regions have 2.0× weight to improve detection accuracy for these challenging regions

## Pretrained Weights

LLRFormer uses HRNet-W32 pretrained on ImageNet as the backbone. Before training, you need to download the pretrained weights:

1. **Download HRNet-W32 ImageNet pretrained weights**:
   - Official HRNet repository: [https://github.com/HRNet/HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)
   - Direct download link: [hrnet_w32-36af842e.pth](https://github.com/HRNet/HRNet-Image-Classification/releases/download/v1.0/hrnet_w32-36af842e.pth)

2. **Place the downloaded file**:
   ```bash
   mkdir -p pretrained/imagenet
   mv hrnet_w32-36af842e.pth pretrained/imagenet/
   ```

3. **Verify the path** in `configs/llrformer.yaml`:
   ```yaml
   PRETRAINED: 'pretrained/imagenet/hrnet_w32-36af842e.pth'
   ```

**Note**: If the pretrained weights file is not found, training will fail with an error message. The pretrained weights are essential for good performance, especially when training data is limited.

## Requirements

- PyTorch >= 1.8.0
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full dependencies
