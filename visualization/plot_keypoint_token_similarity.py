import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import torch

from lib.models.llrformer import get_pose_net
from lib.config import cfg, update_config

CONFIG_FILE = '../configs/llrformer.yaml'
MODEL_PATH = '../output/MyKeypointDataset/llrformer/llrformer/model_best.pth'

# Keypoint names (36 keypoints) - consistent with tools/fix_kpt_order.py
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

# Load configuration and model
update_config(cfg, CONFIG_FILE)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file not found: {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = get_pose_net(cfg, is_train=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

# Extract keypoint tokens
with torch.no_grad():
    kp_init = model.transformer.kp_init
    kp_id_emb = model.transformer.kp_id_emb.weight
    kp_pos = model.transformer.kp_pos
    
    keypoint_token = (kp_init + kp_id_emb.unsqueeze(0) + kp_pos).squeeze(0)
    keypoint_token = keypoint_token.detach().cpu().numpy()
    
    if keypoint_token.shape[0] != len(keypoint_names):
        print(f"Warning: Model has {keypoint_token.shape[0]} keypoints, expected {len(keypoint_names)}")
        if keypoint_token.shape[0] > len(keypoint_names):
            keypoint_token = keypoint_token[:len(keypoint_names), :]
            print(f"Using first {len(keypoint_names)} keypoints")
        else:
            raise ValueError(f"Model has fewer keypoints ({keypoint_token.shape[0]}) than expected ({len(keypoint_names)})")

# Calculate similarity matrix
d = keypoint_token.shape[1]
inner_product = np.matmul(keypoint_token, keypoint_token.T) / np.sqrt(d)

def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)
sim_matrix = softmax(inner_product, axis=0)

# Visualize similarity matrix
plt.figure(figsize=(16, 20))
plt.imshow(sim_matrix, cmap='viridis')
plt.xticks(ticks=np.arange(len(keypoint_names)), labels=keypoint_names, rotation=90)
plt.yticks(ticks=np.arange(len(keypoint_names)), labels=keypoint_names)
cbar = plt.colorbar(label='Similarity', shrink=0.8, aspect=30)
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
os.makedirs('../attention_vis', exist_ok=True)
plt.savefig('../attention_vis/keypoint_token_similarity.png', dpi=200)
plt.show()
print("Successfully saved: ../attention_vis/keypoint_token_similarity.png")
