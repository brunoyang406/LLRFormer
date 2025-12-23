#!/usr/bin/env python3
"""
DataLoader Factory

This module provides functions to create DataLoader instances for training,
validation, and testing. It centralizes data loading configuration and
preprocessing transforms.

Usage:
    from lib.dataset.dataloader import get_train_dataloader, get_val_dataloader
    
    train_loader = get_train_dataloader(cfg)
    val_loader = get_val_dataloader(cfg)
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import dataset
except ImportError:
    # If called directly, import from current package
    from . import mydataset as dataset


def get_train_dataloader(cfg):
    """
    Create DataLoader for training set.
    
    Args:
        cfg: Configuration object containing dataset and training parameters
        
    Returns:
        DataLoader: Training DataLoader instance
    """
    # ImageNet-style normalization for 3-channel RGB input
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create training dataset
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        train_transform
    )
    
    # Calculate batch size: multiply by number of GPUs (or 1 for CPU)
    batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * max(len(cfg.GPUS), 1)
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    return train_loader


def get_val_dataloader(cfg):
    """
    Create DataLoader for validation set.
    
    Args:
        cfg: Configuration object containing dataset and test parameters
        
    Returns:
        DataLoader: Validation DataLoader instance
    """
    # ImageNet-style normalization
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create validation dataset
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        valid_transform
    )
    
    # Calculate batch size: multiply by number of GPUs (or 1 for CPU)
    valid_batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * max(len(cfg.GPUS), 1)
    
    # Create DataLoader
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    
    return valid_loader


def get_test_dataloader(cfg):
    """
    Create DataLoader for test set.
    
    Args:
        cfg: Configuration object containing dataset and test parameters
        
    Returns:
        DataLoader: Test DataLoader instance
    """
    # ImageNet-style normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    # Create test dataset
    test_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    # Calculate batch size: multiply by number of GPUs (or 1 for CPU)
    batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * max(len(cfg.GPUS), 1)
    
    # Create DataLoader (pin_memory only enabled when GPUs are used)
    pin_memory = cfg.PIN_MEMORY if len(cfg.GPUS) > 0 else False
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=pin_memory
    )
    
    return test_loader

