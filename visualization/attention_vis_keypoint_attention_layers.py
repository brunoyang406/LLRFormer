import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from lib.config import default as config_mod
from lib.models.llrformer import get_pose_net
from dataset.mydataset import MyKeypointDataset
from yacs.config import CfgNode as CN

# Load configuration
with open('../configs/llrformer.yaml', 'r', encoding='utf-8') as f:
    cfg_dict = yaml.safe_load(f)
cfg = config_mod._C.clone()
cfg.merge_from_other_cfg(CN(cfg_dict))
cfg.defrost()
cfg.TEST.MODEL_FILE = '../output/MyKeypointDataset/llrformer/llrformer/model_best.pth'
cfg.freeze()

# Load model
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    state_dict = torch.load(cfg.TEST.MODEL_FILE, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully")
except Exception as e:
    print(f"Error: Model loading failed: {e}")
    raise

model.eval()
model = model.to(device)

# Load image
dataset = MyKeypointDataset(cfg, root='../data', set_name='val', is_train=False)
if len(dataset.samples) == 0:
    dataset = MyKeypointDataset(cfg, root='../data', set_name='test', is_train=False)
if len(dataset.samples) == 0:
    raise ValueError("No images found in dataset")

img_idx = 0
img, _, meta = dataset[img_idx]
print(f"Using image: {dataset.samples[img_idx][0]}")
if isinstance(img, np.ndarray):
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
else:
    img_tensor = img.unsqueeze(0).float().to(device) / 255.0

# Register attention hooks
attn_maps = []
def get_hook(attn_maps):
    def hook_fn(m, inp, out):
        if hasattr(m, 'last_attn'):
            attn_maps.append(m.last_attn.detach().cpu())
    return hook_fn

hook_count = 0
# Cross-Self Attention architecture: each layer_group contains [CrossSelfTransformer, Transformer]
for layer_group in model.transformer.cross_self_transformer_layers:
    transformer_layer = layer_group[1]
    if hasattr(transformer_layer, 'layers'):
        for layer in transformer_layer.layers:
            attn_layer = layer[0]
            if hasattr(attn_layer, 'fn') and hasattr(attn_layer.fn, 'fn'):
                attn_module = attn_layer.fn.fn
                attn_module.register_forward_hook(get_hook(attn_maps))
                hook_count += 1

print(f"Registered {hook_count} attention hooks")

# Forward inference
with torch.no_grad():
    _ = model(img_tensor)

print("attn_maps collected:", len(attn_maps))
if len(attn_maps) > 0:
    print("attn_maps[0] shape:", attn_maps[0].shape)
else:
    print("attn_maps is empty!")

# Visualize cross-keypoint attention matrix
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
num_kpt = len(keypoint_names)
layers_to_plot = [0, 1, 2, 3, 4, 5]

plt.figure(figsize=(18, 10))
for i, l in enumerate(layers_to_plot):
    attn = attn_maps[l][0].numpy()
    attn_kpt = attn[:, :num_kpt, :num_kpt].mean(axis=0)

    plt.subplot(2, 3, i+1)
    plt.imshow(attn_kpt, cmap='plasma', vmin=0, vmax=attn_kpt.max())
    plt.xticks(np.arange(num_kpt), keypoint_names, rotation=90, fontsize=6)
    plt.yticks(np.arange(num_kpt), keypoint_names, fontsize=6)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f'Layer #{l+1}')

plt.tight_layout()
os.makedirs('../attention_vis', exist_ok=True)
plt.savefig('../attention_vis/cross_keypoint_attention_layers.png', dpi=200)
plt.show()
print("Successfully saved: ../attention_vis/cross_keypoint_attention_layers.png")
