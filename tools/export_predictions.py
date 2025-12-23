#!/usr/bin/env python3
"""
Export LLRFormer predictions to LabelMe JSON format

This script exports model predictions to LabelMe JSON format for use with 
biomechanics measurement scripts (e.g., measure_new.py).

Usage:
    python tools/export_predictions.py --cfg configs/llrformer.yaml --model-file output/MyKeypointDataset/llrformer/model_best.pth --output-dir output/predictions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.backends.cudnn as cudnn

import _init_paths
from lib.config import default as config_mod
from lib.config.default import update_config
from lib.core.inference import get_final_preds
from dataset.dataloader import get_test_dataloader
import lib.models as models

# Reference keypoint order - consistent with tools/fix_kpt_order.py
REF_ORDER = [
    "R_FC", "L_FC", "R_GT", "L_GT", "R_FNeck_Cut_Up", "L_FNeck_Cut_Up", 
    "R_FNeck_Cut_Down", "L_FNeck_Cut_Down",
    "R_Cdy_Up", "L_Cdy_Up", "R_Cdy_Down", "L_Cdy_Down",
    "R_IF", "L_IF", "R_LLP", "L_LLP", "R_MLP", "L_MLP", 
    "R_LPC", "L_LPC", "R_MPC", "L_MPC",
    "R_IR", "L_IR", "R_LE", "L_LE", "R_ME", "L_ME",
    "R_Cyd_Up", "L_Cyd_Up", "R_Cyd_Down", "L_Cyd_Down",
    "R_DLP", "L_DLP", "R_DMP", "L_DMP"
]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Export predictions to LabelMe JSON format')
    
    parser.add_argument('--cfg',
                        help='Experiment configuration file path',
                        required=True,
                        type=str)
    parser.add_argument('--model-file',
                        help='Path to model weights file',
                        type=str,
                        default='')
    parser.add_argument('--output-dir',
                        help='Output directory for JSON files',
                        type=str,
                        default='output/predictions')
    parser.add_argument('--gpus',
                        help='GPU IDs to use',
                        type=str,
                        default='0')
    parser.add_argument('opts',
                        help="Modify config options",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    return parser.parse_args()


def preds_to_labelme_json(preds, image_path, image_size, keypoint_names):
    """
    Convert predictions to LabelMe JSON format.
    
    Args:
        preds: numpy array of shape [num_keypoints, 2] with (x, y) coordinates
        image_path: path to the image file
        image_size: tuple (width, height) of original image size
        keypoint_names: list of keypoint names in order
        
    Returns:
        dict: LabelMe JSON format dictionary
    """
    shapes = []
    
    for i, (x, y) in enumerate(preds):
        if i >= len(keypoint_names):
            break
            
        label = keypoint_names[i]
        
        shape = {
            "label": label,
            "points": [[float(x), float(y)]],
            "group_id": None,
            "shape_type": "point",
            "flags": {}
        }
        shapes.append(shape)
    
    image_filename = os.path.basename(image_path)
    
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(image_size[1]),
        "imageWidth": int(image_size[0])
    }
    
    return labelme_json


def main():
    args = parse_args()
    cfg = config_mod._C.clone()
    update_config(cfg, args)
    
    if args.model_file:
        cfg.defrost()
        cfg.TEST.MODEL_FILE = args.model_file
        cfg.freeze()
    
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    
    if cfg.TEST.MODEL_FILE:
        print(f'Loading model from {cfg.TEST.MODEL_FILE}')
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location='cpu'), strict=False)
    else:
        raise ValueError("Model file not specified. Use --model-file or set TEST.MODEL_FILE in config.")
    
    if len(cfg.GPUS) > 0:
        print(f"Using GPU: {cfg.GPUS}")
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    else:
        print("Using CPU")
    
    model.eval()
    
    test_loader = get_test_dataloader(cfg)
    test_dataset = test_loader.dataset
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(test_dataset)}")
    print("=" * 60)
    
    idx = 0
    with torch.no_grad():
        for i, (input, target, meta) in enumerate(test_loader):
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs
            
            num_images = input.size(0)
            
            def to_numpy2d(meta_list):
                arr = []
                for x in meta_list:
                    if hasattr(x, 'cpu'):
                        x = x.cpu().numpy()
                    x = np.array(x)
                    arr.append(x.reshape(-1).tolist())
                return np.array(arr, dtype=np.float32)
            
            c = to_numpy2d(meta['center'])
            s = to_numpy2d(meta['scale'])
            c = np.array(c)
            if c.shape != (num_images, 2):
                c = c.reshape(num_images, 2)
            s = np.array(s)
            if s.shape != (num_images, 2):
                s = s.reshape(num_images, 2)
            
            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), c, s)
            
            for j in range(num_images):
                image_path = meta['image'][j]
                
                try:
                    img = Image.open(image_path)
                    img_width, img_height = img.size
                except:
                    img_width = cfg.MODEL.IMAGE_SIZE[0]
                    img_height = cfg.MODEL.IMAGE_SIZE[1]
                
                image_preds = preds[j]
                
                labelme_json = preds_to_labelme_json(
                    image_preds, 
                    image_path, 
                    (img_width, img_height),
                    REF_ORDER
                )
                
                image_filename = os.path.basename(image_path)
                json_filename = os.path.splitext(image_filename)[0] + '.json'
                json_path = output_dir / json_filename
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(labelme_json, f, indent=2, ensure_ascii=False)
                
                idx += 1
                if idx % 10 == 0:
                    print(f"Processed {idx}/{len(test_dataset)} images...")
    
    print("=" * 60)
    print(f"Export completed!")
    print(f"Total exported: {idx} JSON files")
    print(f"Output directory: {output_dir}")
    print("\nYou can now use these JSON files with measure_new.py:")
    print(f"  python measure_new.py {output_dir}")


if __name__ == '__main__':
    main()
