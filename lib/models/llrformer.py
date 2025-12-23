from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import math
from .transformer_backbone import KeypointTransformer
from .hr_base import HRNET_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class LLRFormer(nn.Module):
    """LLRFormer model combining HRNet backbone with KeypointTransformer."""
    
    def __init__(self, cfg, **kwargs):
        super(LLRFormer, self).__init__()
        
        self.pre_feature = HRNET_base(cfg,**kwargs)
        feature_h = 288
        feature_w = 96
        
        patch_h, patch_w = cfg.MODEL.PATCH_SIZE[0], cfg.MODEL.PATCH_SIZE[1]
        feature_h = (feature_h // patch_h) * patch_h
        feature_w = (feature_w // patch_w) * patch_w
        
        use_cross_self_attention = getattr(cfg.MODEL, 'USE_CROSS_SELF_ATTENTION', False)
        cross_self_attention_layers = getattr(cfg.MODEL, 'CROSS_SELF_ATTENTION_LAYERS', 6)
        
        self.transformer = KeypointTransformer(
            feature_size=[feature_h, feature_w],
            patch_size=[patch_h, patch_w],
            num_keypoints=cfg.MODEL.NUM_JOINTS,
            dim=cfg.MODEL.DIM,
            channels=32,
            depth=cfg.MODEL.TRANSFORMER_DEPTH,
            heads=cfg.MODEL.TRANSFORMER_HEADS,
            mlp_dim = cfg.MODEL.DIM*cfg.MODEL.TRANSFORMER_MLP_RATIO,
            apply_init=cfg.MODEL.INIT,
            hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0]//8,
            heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0],
            heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1],cfg.MODEL.HEATMAP_SIZE[0]],
            pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE,
            use_cross_self_attention=use_cross_self_attention,
            cross_self_attention_layers=cross_self_attention_layers)

    def forward(self, x):
        x = self.pre_feature(x)
        x = self.transformer(x)
        return x

    def init_weights(self, pretrained=''):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    """Create and initialize LLRFormer model."""
    model = LLRFormer(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model

