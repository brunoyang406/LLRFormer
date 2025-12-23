"""Transformer Backbone for Keypoint Detection with Cross-Self Attention mechanism."""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
try:
    from timm.models.layers.weight_init import trunc_normal_
except ImportError:
    try:
        from timm.layers.weight_init import trunc_normal_  
    except ImportError:
        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            """Truncated normal initialization."""
            import torch.nn.functional as F
            def norm_cdf(x):
                return (1. + F.erf(x / math.sqrt(2.))) / 2.
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor
import math

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1


def create_zigzag_indices(h, w):
    indices = []
    for i in range(h + w - 1):
        if i % 2 == 0:
            for j in range(min(i + 1, h)):
                if i - j < w:
                    indices.append((i - j, j))
        else:
            for j in range(min(i + 1, w)):
                if i - j < h:
                    indices.append((j, i - j))

    valid_indices = []
    seen = set()
    for row, col in indices:
        if 0 <= row < h and 0 <= col < w and (row, col) not in seen:
            valid_indices.append((row, col))
            seen.add((row, col))

    for row in range(h):
        for col in range(w):
            if (row, col) not in seen:
                valid_indices.append((row, col))

    return valid_indices


def zigzag_rearrange(x, patch_size):
    """Rearrange image patches in zigzag order."""
    B, C, H, W = x.shape
    patch_h, patch_w = patch_size
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    zigzag_indices = create_zigzag_indices(num_patches_h, num_patches_w)

    patches = []
    for h_idx, w_idx in zigzag_indices:
        start_h = h_idx * patch_h
        end_h = start_h + patch_h
        start_w = w_idx * patch_w
        end_w = start_w + patch_w
        patch = x[:, :, start_h:end_h, start_w:end_w]
        patches.append(patch.flatten(1))

    return torch.stack(patches, dim=1)


class Residual(nn.Module):
    """Residual connection wrapper."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """Pre-normalization wrapper."""
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, dim, heads=8, dropout=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        self.last_attn = attn
        return out


class Transformer(nn.Module):
    """Transformer encoder with self-attention and feed-forward layers."""
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None, pos=None):
        for idx, (attn, ff) in enumerate(self.layers):
            if idx > 0 and self.all_attn:
                x[:, self.num_keypoints:] += pos
            x = attn(x, mask=mask)
            x = ff(x)
        return x
    
class CrossAttention(nn.Module):
    """Cross-attention mechanism for keypoint-visual token interaction."""
    def __init__(self, dim, heads=8, dropout=0., scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if not scale_with_head else 1.0 / (dim ** 0.5)

        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, q_inp, kv_inp, mask=None):
        b, nq, d = q_inp.shape
        _, nkv, _ = kv_inp.shape
        h = self.heads

        kv_norm = self.norm_kv(kv_inp)

        q = self.to_q(q_inp).view(b, nq, h, d // h).transpose(1, 2)
        k = self.to_k(kv_norm).view(b, nkv, h, d // h).transpose(1, 2)
        v = self.to_v(kv_norm).view(b, nkv, h, d // h).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :].to(torch.bool)
            dots.masked_fill_(~mask, float('-inf'))

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, nq, d)
        return self.to_out(out)
    
    
class CrossSelfTransformer(nn.Module):
    """Cross-self attention transformer for keypoint-visual token interaction."""
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_keypoints = num_keypoints

        for _ in range(depth):
            self.layers.append(
                Residual(PreNorm(dim, CrossAttention(dim, heads=heads, dropout=dropout, scale_with_head=scale_with_head)))
            )

    def forward(self, x, mask=None, pos=None):
        q = x[:, :self.num_keypoints]
        kv = x[:, self.num_keypoints:]

        for attn in self.layers:
            q_norm = attn.fn.norm(q)
            q_attn = attn.fn.fn(q_norm, kv, mask)
            q = q_attn + q

        out = torch.cat([q, x[:, self.num_keypoints:].detach()], dim=1)
        return out


class KeypointTransformer(nn.Module):
    """Keypoint detection transformer with cross-self attention support."""
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads, mlp_dim,
                 apply_init=False, apply_multi=True, hidden_heatmap_dim=64*6, heatmap_dim=64*48,
                 heatmap_size=[64, 48], channels=3, dropout=0., emb_dropout=0., pos_embedding_type="learnable",
                 use_cross_self_attention=False, cross_self_attention_layers=6):
        super().__init__()
        assert isinstance(feature_size, list) and isinstance(patch_size, list), 'feature_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, 'Feature dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine', 'learnable', 'sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        self.use_cross_self_attention = use_cross_self_attention
        self.cross_self_attention_layers = cross_self_attention_layers

        self.kp_init = nn.Parameter(torch.randn(1, self.num_keypoints, dim) * 0.02)
        self.kp_id_emb = nn.Embedding(self.num_keypoints, dim)
        self.kp_pos = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        trunc_normal_(self.kp_pos, std=0.01)

        h, w = feature_size[0] // patch_size[0], feature_size[1] // patch_size[1]
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        if use_cross_self_attention:
            self.cross_self_transformer_layers = nn.ModuleList([])
            for i in range(cross_self_attention_layers):
                layer = nn.ModuleList([
                    CrossSelfTransformer(
                        dim=dim, depth=1, heads=heads, mlp_dim=mlp_dim, 
                        dropout=dropout, num_keypoints=num_keypoints, scale_with_head=True
                    ),
                    Transformer(
                        dim=dim, depth=1, heads=heads, mlp_dim=mlp_dim, 
                        dropout=dropout, num_keypoints=num_keypoints, 
                        all_attn=self.all_attn, scale_with_head=True
                    )
                ])
                self.cross_self_transformer_layers.append(layer)
        else:
            self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, 
                                         num_keypoints=num_keypoints, all_attn=self.all_attn, 
                                         scale_with_head=True)

        self.to_keypoint_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim <= hidden_heatmap_dim * 0.5 and apply_multi) else nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, heatmap_dim)
        )

        trunc_normal_(self.kp_init, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.vis_pos = None
            self.pos_embedding_full = None
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding_full = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding_full, std=.02)
                self.kp_pos = nn.Parameter(self.pos_embedding_full[:, :self.num_keypoints])
                self.vis_pos = self.pos_embedding_full[:, self.num_keypoints:]
            else:
                self.pos_embedding_full = None
                self.vis_pos = nn.Parameter(self._make_sine_position_embedding(d_model), requires_grad=False)

    def _make_sine_position_embedding(self, d_model, temperature=10000, scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask=None):
        p = self.patch_size
        vis = zigzag_rearrange(feature, p)
        vis = self.patch_to_embedding(vis)
        b, n_vis, _ = vis.shape

        ids = torch.arange(self.num_keypoints, device=vis.device)
        kp = self.kp_init.expand(b, -1, -1) + self.kp_id_emb(ids).unsqueeze(0) + self.kp_pos

        if self.use_cross_self_attention:
            vis_with_pos = vis
            if hasattr(self, 'vis_pos') and self.vis_pos is not None:
                vis_with_pos = vis + self.vis_pos[:, :n_vis]
            
            x = torch.cat((kp, vis_with_pos), dim=1)
            x = self.dropout(x)
            
            for i, (cross_self_transformer, transformer) in enumerate(self.cross_self_transformer_layers):
                x = cross_self_transformer(x, mask, None)
                x = transformer(x, mask, None)
            
        else:
            if hasattr(self, 'vis_pos') and self.vis_pos is not None:
                vis = vis + self.vis_pos[:, :n_vis]

            seq = torch.cat((kp, vis), dim=1)
            seq = self.dropout(seq)

            x = self.transformer(seq, mask, None)

        kp_out = self.to_keypoint_token(x[:, 0:self.num_keypoints])
        x = self.mlp_head(kp_out)
        x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        return x

