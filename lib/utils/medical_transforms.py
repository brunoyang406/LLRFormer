import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class MedicalImageTransforms:
    """Data augmentation strategy for medical images."""
    
    def __init__(self, is_train=True, image_size=(384, 1152)):
        self.is_train = is_train
        self.image_size = image_size
        
        if is_train:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.15,
                    brightness_by_max=True,
                    p=0.3
                ),
                A.OneOf([
                    A.GaussNoise(p=0.2),
                    A.ISONoise(color_shift=(0.005, 0.02), p=0.2),
                    A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.2),
                ], p=0.2),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                    A.MotionBlur(blur_limit=5, p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.2),
                    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.2),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=0.3
                    ),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                ], p=0.5),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    p=0.2
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    p=0.2
                ),
                A.OpticalDistortion(
                    distort_limit=0.2,
                    p=0.2
                ),
                A.OneOf([
                    A.CoarseDropout(
                        p=0.3
                    ),
                    A.GridDropout(
                        ratio=0.3,
                        p=0.3
                    ),
                ], p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
    
    def __call__(self, image, keypoints=None):
        if keypoints is not None:
            keypoints_list = []
            for kp in keypoints:
                if not np.all(kp == 0):
                    keypoints_list.append(kp)
            
            class_labels = [0] * len(keypoints_list)
            
            transformed = self.transform(
                image=image,
                keypoints=keypoints_list,
                class_labels=class_labels
            )
            
            transformed_keypoints = np.zeros_like(keypoints)
            valid_count = 0
            for i, kp in enumerate(keypoints):
                if not np.all(kp == 0):
                    if valid_count < len(transformed['keypoints']):
                        transformed_keypoints[i] = transformed['keypoints'][valid_count]
                        valid_count += 1
            
            return transformed['image'], transformed_keypoints
        else:
            transformed = self.transform(image=image)
            return transformed['image']

class MedicalMixupCutmix:
    """Mixup and Cutmix strategies for medical images."""
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
    
    def mixup(self, images, targets, alpha=0.2):
        """Apply Mixup data augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).cuda()
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_targets = lam * targets + (1 - lam) * targets[index, :]
        
        return mixed_images, mixed_targets
    
    def cutmix(self, images, targets, alpha=1.0):
        """Apply Cutmix data augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).cuda()
        
        y1, x1, y2, x2 = self.rand_bbox(images.size(), lam)
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size()[-1] * images.size()[-2]))
        targets = lam * targets + (1 - lam) * targets[index, :]
        
        return images, targets
    
    def rand_bbox(self, size, lam):
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, images, targets):
        """Apply Mixup or Cutmix augmentation."""
        if np.random.random() < self.cutmix_prob:
            return self.cutmix(images, targets, self.cutmix_alpha)
        else:
            return self.mixup(images, targets, self.mixup_alpha)

class MedicalFocalLoss:
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def __call__(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MedicalLabelSmoothing:
    """Label smoothing for medical images."""
    
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
    
    def __call__(self, targets, num_classes):
        with torch.no_grad():
            targets = targets * (1 - self.smoothing) + self.smoothing / num_classes
        return targets