# Visualization Scripts

This directory contains scripts for visualizing and analyzing the attention mechanisms and keypoint token similarities in the LLRFormer model.

## Scripts Overview

### 1. `attention_vis_keypoint_attention_layers.py`
Visualizes cross-keypoint attention matrices across all transformer layers.

**Usage:**
```bash
cd visualization
python attention_vis_keypoint_attention_layers.py
```

**Output:**
- `../attention_vis/cross_keypoint_attention_layers.png` - Attention heatmaps for all 6 layers

### 2. `attention_visualize.py`
Comprehensive attention visualization script that generates:
- Cross-keypoint attention matrices (keypoint-to-keypoint dependencies)
- Keypoint-to-image attention maps (how keypoints attend to image patches)
- Symmetry analysis for left-right symmetric keypoints

**Usage:**
```bash
cd visualization
python attention_visualize.py
```

**Output:**
- `../attention_vis/cross_keypoint_attention_layers.jpg` - Cross-keypoint attention for all layers
- `../attention_vis/keypoint_to_image_attention_keypoints_4_5_14_15.jpg` - Keypoint-to-image attention visualization
- Console output with symmetry analysis results

**Note:** The script automatically searches for test images in `../data/test/`. Update the default image path in the script if needed.

### 3. `plot_keypoint_token_similarity.py`
Visualizes keypoint token similarity as a heatmap matrix.

**Usage:**
```bash
cd visualization
python plot_keypoint_token_similarity.py
```

**Output:**
- `../attention_vis/keypoint_token_similarity.png` - Heatmap visualization of keypoint token similarities

## Requirements

All scripts require:
- Trained LLRFormer model weights (scripts will automatically search for model files in `../output/MyKeypointDataset/llrformer/`)
- Configuration file at `../configs/llrformer.yaml`
- Data directory at `../data/` (for `attention_visualize.py`)

## Output Directory

All visualization outputs are saved to `../attention_vis/` (relative to this directory, which is `attention_vis/` in the project root).

## Keypoint Names

All scripts use the standard 36-keypoint naming convention consistent with `tools/fix_kpt_order.py`:
- R_FC, L_FC, R_GT, L_GT, R_FNeck_Cut_Up, L_FNeck_Cut_Up, R_FNeck_Cut_Down, L_FNeck_Cut_Down
- R_Cdy_Up, L_Cdy_Up, R_Cdy_Down, L_Cdy_Down
- R_IF, L_IF, R_LLP, L_LLP, R_MLP, L_MLP, R_LPC, L_LPC, R_MPC, L_MPC
- R_IR, L_IR, R_LE, L_LE, R_ME, L_ME
- R_Cyd_Up, L_Cyd_Up, R_Cyd_Down, L_Cyd_Down
- R_DLP, L_DLP, R_DMP, L_DMP

## Notes

- All scripts use relative paths (`../`) to access project root directories
- Scripts should be run from the `visualization/` directory
- Make sure you have a trained model before running visualization scripts
- Some scripts require GPU for faster execution, but will fall back to CPU if GPU is not available

