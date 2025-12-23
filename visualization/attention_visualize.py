import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import yaml

from lib.config import default as config_mod
from lib.models.llrformer import get_pose_net
from dataset.mydataset import MyKeypointDataset
from yacs.config import CfgNode as CN

# Keypoint order (36 keypoints) - consistent with tools/fix_kpt_order.py
keypoint_names = [
    "R_FC", "L_FC", "R_GT", "L_GT", "R_FNeck_Cut_Up", "L_FNeck_Cut_Up", 
    "R_FNeck_Cut_Down", "L_FNeck_Cut_Down",
    "R_Cdy_Up", "L_Cdy_Up", "R_Cdy_Down", "L_Cdy_Down",
    "R_IF", "L_IF", "R_LLP", "L_LLP", "R_MLP", "L_MLP", 
    "R_LPC", "L_LPC", "R_MPC", "L_MPC",
    "R_IR", "L_IR", "R_LE", "L_LE", "R_ME", "L_ME",
    "R_Cyd_Up", "L_Cyd_Up", "R_Cyd_Down", "L_Cyd_Down",
    "R_DLP", "L_DLP", "R_DMP", "L_DMP"
]

# Load configuration
with open('../configs/llrformer.yaml', 'r', encoding='utf-8') as f:
    cfg_dict = yaml.safe_load(f)
cfg = config_mod._C.clone()
cfg.merge_from_other_cfg(CN(cfg_dict))
cfg.defrost()
cfg.TEST.MODEL_FILE = '../output/MyKeypointDataset/llrformer/llrformer/model_best.pth'
cfg.freeze()

print("Using configuration: llrformer.yaml")
print(f"Image size: {cfg.MODEL.IMAGE_SIZE}")
print(f"Heatmap size: {cfg.MODEL.HEATMAP_SIZE}")
print(f"Patch size: {cfg.MODEL.PATCH_SIZE}")
print(f"Transformer depth: {cfg.MODEL.TRANSFORMER_DEPTH}")
print(f"Attention heads: {cfg.MODEL.TRANSFORMER_HEADS}")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = get_pose_net(cfg, is_train=False)

if not os.path.exists(cfg.TEST.MODEL_FILE):
    print(f"Warning: Model file not found: {cfg.TEST.MODEL_FILE}")
    possible_paths = [
        '../output/MyKeypointDataset/llrformer/llrformer/model_best.pth',
        '../output/MyKeypointDataset/llrformer/model_best.pth',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            cfg.TEST.MODEL_FILE = path
            print(f"Found available model file: {path}")
            break
    else:
        raise FileNotFoundError("No available model file found")

print(f"Loading model file: {cfg.TEST.MODEL_FILE}")

try:
    state_dict = torch.load(cfg.TEST.MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully")
except Exception as e:
    print(f"Error: Model loading failed: {e}")
    raise

model.eval()
model = model.to(device)
print("Model ready")

# Load image and annotation
default_image = r'../data/test/images/1_2_410_200049_2_47176438882121_3_1_20180409083633651_83807.jpeg'
default_annotation = r'../data/test/annotations/1_2_410_200049_2_47176438882121_3_1_20180409083633651_83807.json'

if os.path.exists(default_image) and os.path.exists(default_annotation):
    image_path = default_image
    annotation_path = default_annotation
else:
    # Find first available test file
    test_ann_dir = '../data/test/annotations'
    test_img_dir = '../data/test/images'
    found = False
    
    if os.path.exists(test_ann_dir):
        for json_file in os.listdir(test_ann_dir):
            if json_file.endswith('.json'):
                base_name = os.path.splitext(json_file)[0]
                for ext in ['.jpeg', '.jpg', '.png']:
                    img_file = base_name + ext
                    img_path = os.path.join(test_img_dir, img_file)
                    ann_path = os.path.join(test_ann_dir, json_file)
                    if os.path.exists(img_path) and os.path.exists(ann_path):
                        image_path = img_path
                        annotation_path = ann_path
                        found = True
                        print(f"Using test file: {json_file}")
                        break
                if found:
                    break
    
    if not found:
        raise FileNotFoundError(f"Could not find any test image/annotation pairs in {test_img_dir} and {test_ann_dir}")

# Check if files exist
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")
if not os.path.exists(annotation_path):
    raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

print(f"Loading image: {image_path}")
print(f"Loading annotation: {annotation_path}")

# Load image
from PIL import Image
import numpy as np
import json

img_pil = Image.open(image_path).convert('RGB')
img = np.array(img_pil)
img_vis = img.copy()

print(f"Original image size: {img.shape}")

# Load annotation file
with open(annotation_path, 'r', encoding='utf-8') as f:
    annotation = json.load(f)

# Parse keypoints
keypoints = []
for shape in annotation['shapes']:
    if shape['shape_type'] == 'point':
        keypoints.append(shape['points'][0])
    elif shape['shape_type'] == 'circle':
        keypoints.append(shape['points'][0])

keypoints = np.array(keypoints, dtype=np.float32)
print(f"Original keypoint count: {len(keypoints)}")
print(f"Keypoint coordinate range: X[{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}], Y[{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")

# Resize and pad image to target size (384x1152)
target_w, target_h = 384, 1152
orig_h, orig_w = img.shape[:2]

# Calculate scale factor
scale_w = target_w / orig_w
scale_h = target_h / orig_h
scale = min(scale_w, scale_h)

# Calculate resized dimensions
new_w, new_h = int(orig_w * scale), int(orig_h * scale)
print(f"Target size: {target_w} x {target_h}")
print(f"Scale factor: {scale:.4f}")
print(f"Resized size: {new_w} x {new_h}")

# Resize image
resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# Create blank image with target size
input_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

# Calculate padding position (centered)
pad_left = (target_w - new_w) // 2
pad_top = (target_h - new_h) // 2

# Place resized image in the center of target image
input_img[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized_img

print(f"Final image size: {input_img.shape}")
print(f"Padding: left={pad_left}, top={pad_top}")

# Transform keypoints accordingly and ensure within boundaries
transformed_keypoints = keypoints.copy()
transformed_keypoints[:, 0] = np.clip(keypoints[:, 0] * scale + pad_left, 0, target_w - 1)
transformed_keypoints[:, 1] = np.clip(keypoints[:, 1] * scale + pad_top, 0, target_h - 1)

print(f"Transformed keypoint coordinate range: X[{transformed_keypoints[:, 0].min():.1f}, {transformed_keypoints[:, 0].max():.1f}], Y[{transformed_keypoints[:, 1].min():.1f}, {transformed_keypoints[:, 1].max():.1f}]")

# Convert to tensor
img_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

query_locations = transformed_keypoints
print(f"Using ground truth keypoint locations, count: {len(query_locations)}")

target_kpts = [4, 5, 14, 15]
for kpt_id in target_kpts:
    if kpt_id < len(query_locations):
        x, y = query_locations[kpt_id]
        print(f"Keypoint {kpt_id} ({keypoint_names[kpt_id]}): position ({x:.1f}, {y:.1f})")
    else:
        print(f"Warning: Keypoint {kpt_id} out of range, total keypoints: {len(query_locations)}")

# Register attention hooks
attn_maps = []

def register_attention_hooks(model):
    """Register hooks for all attention layers"""
    hooks = []
    hook_count = 0
    
    # Cross-Self Attention architecture: each layer_group contains [CrossSelfTransformer, Transformer]
    for layer_group_idx, layer_group in enumerate(model.transformer.cross_self_transformer_layers):
        transformer_layer = layer_group[1]
        if hasattr(transformer_layer, 'layers'):
            for layer in transformer_layer.layers:
                attn_layer = layer[0]
                if hasattr(attn_layer, 'fn') and hasattr(attn_layer.fn, 'fn'):
                    attn_module = attn_layer.fn.fn
                    def make_hook(layer_idx):
                        def hook_fn(m, inp, out):
                            if hasattr(m, 'last_attn'):
                                attn_maps.append(m.last_attn.detach().cpu())
                                print(f"Captured Layer {layer_idx+1} Self-Attention")
                        return hook_fn
                    hook = attn_module.register_forward_hook(make_hook(layer_group_idx))
                    hooks.append(hook)
                    hook_count += 1
    
    print(f"Total registered attention hooks: {hook_count}")
    return hooks

hooks = register_attention_hooks(model)

# Forward inference
with torch.no_grad():
    _ = model(img_tensor)

# Visualization functions
def plot_attention_for_specific_keypoints(img, query_locations, attn_maps, kpt_ids, save_prefix="../attention_vis/tokenpose_attention", threshold=0.0, use_blur=True):
    """
    Visualize attention for specific keypoints.
    kpt_ids: List of keypoint IDs, e.g., [4, 5, 14, 15]
    """
    img_vis = img.copy()
    
    # Use heatmap size and patch size from config
    h_patch = cfg.MODEL.HEATMAP_SIZE[0] // cfg.MODEL.PATCH_SIZE[0]   
    w_patch = cfg.MODEL.HEATMAP_SIZE[1] // cfg.MODEL.PATCH_SIZE[1]   
    
    print(f"Patch size: {cfg.MODEL.PATCH_SIZE}")
    print(f"Heatmap size: {cfg.MODEL.HEATMAP_SIZE}")
    print(f"Calculated patch count: {h_patch} x {w_patch}")
    print(f"Captured attention maps count: {len(attn_maps)}")
    
    img_h, img_w = img_vis.shape[0:2]
    n_layers = min(len(attn_maps), cfg.MODEL.TRANSFORMER_DEPTH)
    show_kpt = len(kpt_ids)

    if n_layers == 0:
        print("Error: No attention maps captured, please check model structure and hook registration")
        return

    # Create figure
    fig, axs = plt.subplots(n_layers, show_kpt + 1, figsize=(3 * (show_kpt + 1), 3 * n_layers))
    if n_layers == 1:
        axs = axs.reshape(1, -1)
    
    # Set main title
    fig.suptitle(f'LLRFormer Attention Visualization\nKeypoints: {[keypoint_names[i] for i in kpt_ids]}\nImage Size: {img_w}x{img_h}', 
                 fontsize=16, y=0.98)

    for l in range(n_layers):
        # First column: Original image + all keypoint annotations
        axs[l][0].imshow(img_vis)
        for k in kpt_ids:
            axs[l][0].scatter(query_locations[k, 0], query_locations[k, 1], s=80, marker="*", c="red", edgecolors="white", linewidth=2)
        axs[l][0].set_ylabel(f"Layer {l+1}", fontsize=12, fontweight='bold')
        axs[l][0].set_xticks([]); axs[l][0].set_yticks([])
        axs[l][0].set_title("Original Image", fontsize=10)

        # Other columns: Attention map for each keypoint
        for idx, k in enumerate(kpt_ids):
            if l < len(attn_maps):
                attn = attn_maps[l]
                
                # Handle different attention formats
                if len(attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    attn = attn[0].mean(0)  # Take first batch, average over heads
                elif len(attn.shape) == 3:  # [heads, seq_len, seq_len]
                    attn = attn.mean(0)  # Average over heads
                elif len(attn.shape) == 2:  # [seq_len, seq_len]
                    attn = attn  # Already 2D
                else:
                    print(f"Warning: Unknown attention shape: {attn.shape}")
                    continue
                
                print(f"Layer {l+1} attention shape: {attn.shape}")
                
                # Get attention pattern for keypoint k
                if k < attn.shape[0] and len(query_locations) < attn.shape[1]:
                    # Keypoint to image patches attention
                    patch_token = attn[k, len(query_locations):]
                elif k < attn.shape[0]:
                    # If attention matrix is smaller, use directly
                    patch_token = attn[k, :]
                else:
                    print(f"Warning: Keypoint {k} out of attention matrix range")
                    continue
                
                # Reshape attention map according to actual patch count
                try:
                    patch_map = patch_token.reshape(h_patch, w_patch).numpy()
                except:
                    # If reshape fails, use 1D and adjust
                    total_patches = patch_token.shape[0]
                    h_patch_actual = int(np.sqrt(total_patches))
                    w_patch_actual = total_patches // h_patch_actual
                    if h_patch_actual * w_patch_actual == total_patches:
                        patch_map = patch_token[:h_patch_actual * w_patch_actual].reshape(h_patch_actual, w_patch_actual).numpy()
                        print(f"Warning: Using adjusted patch size: {h_patch_actual} x {w_patch_actual}")
                    else:
                        # If still fails, create 1D heatmap
                        patch_map = patch_token.numpy().reshape(1, -1)
                        print(f"Warning: Using 1D heatmap: {patch_map.shape}")

                # Resize attention map to image size
                if len(patch_map.shape) == 2:
                    patch_resized = cv2.resize(patch_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                else:
                    # 1D case, create simple heatmap
                    patch_resized = np.tile(patch_map.flatten()[:img_h*img_w].reshape(img_h, img_w), (1, 1))
                
                if use_blur and len(patch_resized.shape) == 2:
                    patch_resized = cv2.GaussianBlur(patch_resized, (5, 5), 0)
                
                # Normalize
                if patch_resized.max() > patch_resized.min():
                    patch_resized = (patch_resized - patch_resized.min()) / (patch_resized.max() - patch_resized.min() + 1e-8)
                if threshold > 0:
                    patch_resized[patch_resized <= threshold] = 0

                # Visualize
                axs[l][idx+1].imshow(img_vis)
                im = axs[l][idx+1].imshow(patch_resized, cmap='jet', alpha=0.7)
                axs[l][idx+1].scatter(query_locations[k, 0], query_locations[k, 1], s=80, marker="*", c="red", edgecolors="white", linewidth=2)
                axs[l][idx+1].set_xticks([]); axs[l][idx+1].set_yticks([])
                
                # Set column title
                if l == 0:
                    axs[l][idx+1].set_title(f"{keypoint_names[k]}\n(Kpt {k})", fontsize=10, fontweight='bold')

    # Add colorbar
    if 'im' in locals():
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Attention Weight', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9, top=0.95)
    
    # Save image
    save_path = f"{save_prefix}_keypoints_{'_'.join(map(str, kpt_ids))}.jpg"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization completed: {save_path}")
    print(f"Analyzed keypoints: {[keypoint_names[i] for i in kpt_ids]}")
    print(f"Keypoint locations: {[query_locations[i] for i in kpt_ids]}")
    print(f"Using configuration: llrformer.yaml")
    print(f"Image size: {img_w}x{img_h}")

# Generate attention visualizations
print("\nGenerating two styles of attention maps...")

def visualize_cross_keypoint_attention(attn_maps, save_prefix="attention_vis"):
    print("Generating Cross-keypoint Attention Matrix...")
    
    layers_to_plot = [0, 1, 2, 3, 4, 5]
    
    plt.figure(figsize=(18, 10))
    for i, l in enumerate(layers_to_plot):
        if l < len(attn_maps):
            attn = attn_maps[l][0].numpy()  # [num_heads, num_tokens, num_tokens]
            # Extract attention only between keypoint tokens
            num_kpt = min(len(keypoint_names), attn.shape[1])
            attn_kpt = attn[:, :num_kpt, :num_kpt].mean(axis=0)  # [num_kpt, num_kpt], average over heads
            
            plt.subplot(2, 3, i+1)
            plt.imshow(attn_kpt, cmap='plasma', vmin=0, vmax=attn_kpt.max())
            plt.xticks(np.arange(num_kpt), keypoint_names[:num_kpt], rotation=90, fontsize=6)
            plt.yticks(np.arange(num_kpt), keypoint_names[:num_kpt], fontsize=6)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f'Layer #{l+1}')
    
    plt.tight_layout()
    save_path = f"../{save_prefix}/cross_keypoint_attention_layers.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Cross-keypoint Attention Matrix saved successfully: {save_path}")

def visualize_keypoint_to_image_attention(img, attn_maps, kpt_ids, query_locations, save_prefix="attention_vis"):
    print("Generating Keypoint-to-Image Attention...")
    
    layers_to_plot = [0, 1, 2, 3, 4, 5]
    
    # Calculate patch count
    h_patch = cfg.MODEL.HEATMAP_SIZE[0] // cfg.MODEL.PATCH_SIZE[0]
    w_patch = cfg.MODEL.HEATMAP_SIZE[1] // cfg.MODEL.PATCH_SIZE[1]
    
    # Calculate image size
    img_h, img_w = img.shape[:2]
    
    # Create subplots
    fig, axes = plt.subplots(len(layers_to_plot), len(kpt_ids), 
                            figsize=(4*len(kpt_ids), 4*len(layers_to_plot)))
    
    # If only one row or column, ensure axes is 2D array
    if len(layers_to_plot) == 1:
        axes = axes.reshape(1, -1)
    if len(kpt_ids) == 1:
        axes = axes.reshape(-1, 1)
    
    # Generate attention maps for each keypoint and layer
    for i, layer_idx in enumerate(layers_to_plot):
        if layer_idx >= len(attn_maps):
            continue
            
        for j, kpt_id in enumerate(kpt_ids):
            if kpt_id >= len(query_locations):
                continue
                
            ax = axes[i, j]
            
            # Get attention weights
            attn = attn_maps[layer_idx][0]  # [num_heads, num_tokens, num_tokens]
            
            # Get keypoint token attention to image patches
            kpt_token_idx = kpt_id
            if kpt_token_idx < attn.shape[1]:
                # Keypoint token attention to all patches
                kpt_attn = attn[:, kpt_token_idx, len(query_locations):].mean(dim=0)  # [num_patches]
                
                # Reshape to 2D heatmap
                try:
                    attn_map = kpt_attn.reshape(h_patch, w_patch).numpy()
                except:
                    # If reshape fails, use 1D and adjust
                    total_patches = kpt_attn.shape[0]
                    h_patch_actual = int(np.sqrt(total_patches))
                    w_patch_actual = total_patches // h_patch_actual
                    attn_map = kpt_attn[:h_patch_actual * w_patch_actual].reshape(h_patch_actual, w_patch_actual).numpy()
                    print(f"Warning: Using adjusted patch size: {h_patch_actual} x {w_patch_actual}")
                
                # Upsample to image size
                attn_map_resized = cv2.resize(attn_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                
                # Display original image
                ax.imshow(img, cmap='gray', alpha=0.7)
                
                im = ax.imshow(attn_map_resized, cmap='jet', alpha=0.6, vmin=0, vmax=attn_map_resized.max())
                
                # Mark keypoint position
                x, y = query_locations[kpt_id]
                ax.plot(x, y, 'w*', markersize=10, markeredgecolor='black', markeredgewidth=1)
                
                # Set title and labels
                if i == 0:  # First row shows keypoint names
                    ax.set_title(f'{keypoint_names[kpt_id]}', fontsize=10, fontweight='bold')
                if j == 0:  # First column shows layer numbers
                    ax.set_ylabel(f'Layer {layer_idx+1}', fontsize=10, fontweight='bold')
                
                ax.axis('off')
    
    # Add colorbar (only if im was defined)
    if 'im' in locals():
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=12)
    
    # Set main title
    fig.suptitle(f'LLRFormer Keypoint-to-Image Attention\nKeypoints: {[keypoint_names[i] for i in kpt_ids]}\nImage Size: {img_w}x{img_h}', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save image
    save_path = f"../{save_prefix}/keypoint_to_image_attention_keypoints_{'_'.join(map(str, kpt_ids))}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Keypoint-to-Image Attention saved successfully: {save_path}")
    print(f"Image size: {img_w}x{img_h}")

# For specified keypoints (4,5) and (14,15)
target_keypoints = [4, 5, 14, 15]
print(f"Starting analysis for keypoints: {[keypoint_names[i] for i in target_keypoints]}")

# Generate two styles of plots
visualize_cross_keypoint_attention(attn_maps, "attention_vis")
visualize_keypoint_to_image_attention(input_img, attn_maps, target_keypoints, query_locations, "attention_vis")

print("\nAll attention maps generated successfully!")
print("Save location: ../attention_vis/")
print("   - cross_keypoint_attention_layers.jpg (keypoint-to-keypoint attention dependencies)")
print("   - keypoint_to_image_attention_keypoints_4_5_14_15.jpg (keypoint-to-image attention)")

# Symmetry analysis
def analyze_symmetry_attention(attn_maps, left_kpt, right_kpt, query_locations):
    """Analyze attention patterns for left-right symmetric keypoints"""
    print(f"\nSymmetry analysis: {keypoint_names[left_kpt]} vs {keypoint_names[right_kpt]}")
    
    # Calculate positions of two keypoints
    left_pos = query_locations[left_kpt]
    right_pos = query_locations[right_kpt]
    
    print(f"Left position: {left_pos}")
    print(f"Right position: {right_pos}")
    
    for layer_idx in range(min(len(attn_maps), 6)):
        attn = attn_maps[layer_idx][0].mean(0)
        
        # Get attention patterns for two keypoints
        patch_token_left = attn[left_kpt, len(query_locations):]
        patch_token_right = attn[right_kpt, len(query_locations):]
        
        # Reshape attention map according to actual patch count
        total_patches = patch_token_left.shape[0]
        h_patch_actual = int(np.sqrt(total_patches))
        w_patch_actual = total_patches // h_patch_actual
        
        try:
            left_attn = patch_token_left[:h_patch_actual * w_patch_actual].reshape(h_patch_actual, w_patch_actual)
            right_attn = patch_token_right[:h_patch_actual * w_patch_actual].reshape(h_patch_actual, w_patch_actual)
        except:
            # If reshape fails, use 1D
            left_attn = patch_token_left
            right_attn = patch_token_right
        
        # Calculate similarity (cosine similarity)
        left_flat = left_attn.flatten()
        right_flat = right_attn.flatten()
        
        similarity = np.dot(left_flat, right_flat) / (np.linalg.norm(left_flat) * np.linalg.norm(right_flat) + 1e-8)
        
        if left_kpt in [4, 5]:
            region = "Hip region"
            weight_info = f"(weight: {cfg.TRAIN.HIP_WEIGHT if hasattr(cfg.TRAIN, 'HIP_WEIGHT') else '1.0'})"
        elif left_kpt in [14, 15]:
            region = "Femur region"
            weight_info = f"(weight: {cfg.TRAIN.FEMUR_WEIGHT if hasattr(cfg.TRAIN, 'FEMUR_WEIGHT') else '1.0'})"
        else:
            region = "Other region"
            weight_info = "(weight: 1.0)"
        
        print(f"   Layer {layer_idx+1}: Similarity = {similarity:.4f} | {region} {weight_info}")

# Analyze symmetric keypoint pairs
print("\n" + "="*80)
print("LLRFormer Symmetry Constraint Verification Analysis")
print("="*80)
print(f"Configuration features:")
print(f"   - Keypoint weight strategy: {'Enabled' if getattr(cfg.TRAIN, 'JOINT_WEIGHTS', False) else 'Disabled'}")
print(f"   - Femur/Tibia weight: {cfg.TRAIN.FEMUR_WEIGHT if hasattr(cfg.TRAIN, 'FEMUR_WEIGHT') else '1.0'}")
print(f"   - Attention dropout: {cfg.MODEL.ATTENTION_DROPOUT}")
print(f"   - Position encoding: {cfg.MODEL.POS_EMBEDDING_TYPE}")
print("="*80)

if 4 in target_keypoints and 5 in target_keypoints:
    print(f"\nHip symmetry analysis (keypoints 4-5)")
    print(f"   Keypoint 4: {keypoint_names[4]} (Right greater trochanter)")
    print(f"   Keypoint 5: {keypoint_names[5]} (Left greater trochanter)")
    analyze_symmetry_attention(attn_maps, 4, 5, query_locations)

if 14 in target_keypoints and 15 in target_keypoints:
    print(f"\nFemur symmetry analysis (keypoints 14-15)")
    print(f"   Keypoint 14: {keypoint_names[14]} (Right medial condyle)")
    print(f"   Keypoint 15: {keypoint_names[15]} (Left medial condyle)")
    analyze_symmetry_attention(attn_maps, 14, 15, query_locations)

print(f"\nAnalysis completed! These visualizations demonstrate LLRFormer's attention mechanism")
print(f"   with special focus on symmetry constraint learning in hip and femur regions")

# Cleanup hooks
def cleanup_hooks(hooks):
    """Remove all registered hooks"""
    for hook in hooks:
        hook.remove()
    print(f"Cleaned up {len(hooks)} hooks")

cleanup_hooks(hooks)
