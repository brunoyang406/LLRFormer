#!/usr/bin/env python3
"""
Keypoint Order Fixer

This script fixes keypoint order in annotation files to match a reference order.
It creates backups before modifying files and verifies the results.

Usage:
    python tools/fix_kpt_order.py
"""

import os
import json
import sys
from glob import glob
import shutil
from datetime import datetime


# Reference keypoint order
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


def fix_file_order(jf, ref_order, backup_dir):
    """
    Fix keypoint order in a single annotation file.
    
    Args:
        jf: JSON file path
        ref_order: Reference keypoint order list
        backup_dir: Backup directory path
        
    Returns:
        tuple: (success, missing_labels, extra_labels)
    """
    try:
        with open(jf, 'r', encoding='utf-8') as f:
            ann = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read {jf}: {e}")
        return False, [], []
    
    backup_path = os.path.join(backup_dir, os.path.basename(jf))
    shutil.copy2(jf, backup_path)
    
    current_order = [shape['label'] for shape in ann['shapes']]
    
    if current_order == ref_order:
        return True, [], []
    
    label_to_shape = {shape['label']: shape for shape in ann['shapes']}
    
    new_shapes = []
    missing_labels = []
    for label in ref_order:
        if label in label_to_shape:
            new_shapes.append(label_to_shape[label])
        else:
            missing_labels.append(label)
    
    extra_labels = list(set(current_order) - set(ref_order))
    
    ann['shapes'] = new_shapes
    try:
        with open(jf, 'w', encoding='utf-8') as f:
            json.dump(ann, f, indent=2, ensure_ascii=False)
        return True, missing_labels, extra_labels
    except Exception as e:
        print(f"Error: Failed to write {jf}: {e}")
        return False, missing_labels, extra_labels


def verify_fix(ann_dirs, ref_order):
    """
    Verify that all files have been fixed correctly.
    
    Args:
        ann_dirs: List of annotation directory paths
        ref_order: Reference keypoint order list
        
    Returns:
        bool: True if consistent, False otherwise
    """
    all_consistent = True
    for ann_dir in ann_dirs:
        if not os.path.exists(ann_dir):
            continue
        json_files = sorted(glob(os.path.join(ann_dir, '*.json')))
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    ann = json.load(f)
                current_order = [shape['label'] for shape in ann['shapes']]
                if current_order != ref_order:
                    print(f"Inconsistent order: {os.path.basename(jf)} ({ann_dir})")
                    all_consistent = False
            except Exception as e:
                print(f"Error reading {jf}: {e}")
                all_consistent = False
    
    return all_consistent


def main():
    """Main function."""
    ann_dirs = [
        'data/test/annotations',
        'data/train/annotations',
        'data/val/annotations',
        'data/external/external/annotations',
    ]
    
    print("Keypoint Order Fixer")
    print("=" * 60)
    print("Reference order:")
    for i, label in enumerate(REF_ORDER, 1):
        print(f"  {i:2d}. {label}")
    print(f"\nTotal keypoints: {len(REF_ORDER)}\n")
    
    fixed_count = 0
    total_count = 0
    backup_dirs = []
    
    for ann_dir in ann_dirs:
        if not os.path.exists(ann_dir):
            print(f"Warning: Directory not found: {ann_dir}")
            continue
        
        json_files = sorted(glob(os.path.join(ann_dir, '*.json')))
        if len(json_files) == 0:
            print(f"Warning: No JSON files found in {ann_dir}")
            continue
        
        backup_dir = f"{ann_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        backup_dirs.append(backup_dir)
        print(f"Backup directory: {backup_dir}")
        
        for jf in json_files:
            current_order = []
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    ann = json.load(f)
                current_order = [shape['label'] for shape in ann['shapes']]
            except:
                pass
            
            if current_order == REF_ORDER:
                print(f"Correct order: {os.path.basename(jf)}")
                total_count += 1
                continue
            
            print(f"Fixing order: {os.path.basename(jf)}")
            success, missing_labels, extra_labels = fix_file_order(jf, REF_ORDER, backup_dir)
            
            if success:
                if missing_labels:
                    print(f"  Warning: Missing keypoints: {missing_labels}")
                if extra_labels:
                    print(f"  Warning: Extra keypoints: {extra_labels}")
                fixed_count += 1
            else:
                print(f"  Error: Failed to fix {jf}")
            
            total_count += 1
        
        print()
    
    print("=" * 60)
    print("Fix Summary:")
    print(f"  Total files processed: {total_count}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Backup locations: {backup_dirs}")
    print()
    
    print("Verifying fix results...")
    all_consistent = verify_fix(ann_dirs, REF_ORDER)
    
    if all_consistent:
        print("Success: All files have consistent keypoint order!")
        sys.exit(0)
    else:
        print("Warning: Some files still have inconsistent order. Please check manually.")
        sys.exit(1)


if __name__ == '__main__':
    main() 