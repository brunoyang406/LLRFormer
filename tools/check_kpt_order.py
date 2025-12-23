#!/usr/bin/env python3
"""
Keypoint Order Checker

This script checks if all annotation files have consistent keypoint order.
It compares the keypoint label order across all JSON annotation files in the dataset.

Usage:
    python tools/check_kpt_order.py
"""

import os
import json
import sys
from glob import glob


def load_keypoint_orders(ann_dirs):
    """
    Load keypoint orders from all annotation files.
    
    Args:
        ann_dirs: List of annotation directory paths
        
    Returns:
        tuple: (all_orders, file_orders)
    """
    all_orders = []
    file_orders = {}
    
    for ann_dir in ann_dirs:
        if not os.path.exists(ann_dir):
            continue
        json_files = sorted(glob(os.path.join(ann_dir, '*.json')))
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    ann = json.load(f)
                labels = [shape['label'] for shape in ann['shapes']]
                all_orders.append(labels)
                file_orders[jf] = labels
            except Exception as e:
                print(f"Warning: Failed to read {jf}: {e}")
                continue
    
    return all_orders, file_orders


def check_consistency(all_orders, file_orders):
    """
    Check if all files have consistent keypoint order.
    
    Args:
        all_orders: List of all keypoint orders
        file_orders: Dict mapping file path to keypoint order
        
    Returns:
        bool: True if consistent, False otherwise
    """
    if not all_orders:
        print("Error: No JSON files found!")
        return False
    
    ref_order = all_orders[0]
    ref_file = list(file_orders.keys())[0]
    
    print(f"Reference order (from {ref_file}):")
    for i, label in enumerate(ref_order, 1):
        print(f"  {i:2d}. {label}")
    print(f"\nTotal keypoints: {len(ref_order)}\n")
    
    inconsistent_files = []
    for jf, order in file_orders.items():
        if order != ref_order:
            inconsistent_files.append((jf, order))
            print(f"Inconsistent order: {jf}")
            print(f"  Order: {order}")
            print()
        else:
            print(f"Consistent order: {os.path.basename(jf)}")
    
    if inconsistent_files:
        print(f"\nFound {len(inconsistent_files)} file(s) with inconsistent order.")
        return False
    else:
        print(f"\nAll {len(file_orders)} files have consistent keypoint order!")
        return True


def main():
    """Main function."""
    ann_dirs = [
        'data/test/annotations',
        'data/train/annotations',
        'data/val/annotations',
        'data/external/external/annotations',
    ]
    
    print("Checking keypoint order consistency...")
    print("=" * 60)
    
    all_orders, file_orders = load_keypoint_orders(ann_dirs)
    is_consistent = check_consistency(all_orders, file_orders)
    
    print("=" * 60)
    if is_consistent:
        print("Check completed: All files are consistent!")
        sys.exit(0)
    else:
        print("Check completed: Found inconsistent files!")
        print("Use tools/fix_kpt_order.py to fix the order.")
        sys.exit(1)


if __name__ == '__main__':
    main() 