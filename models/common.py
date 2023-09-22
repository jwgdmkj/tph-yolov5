# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import logging
import math
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (colorstr, increment_path, make_divisible, non_max_suppression, save_one_box, scale_coords,
                           xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

from einops import rearrange, repeat
from torch import einsum
from einops.layers.torch import Rearrange

LOGGER = logging.getLogger(__name__)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.sigmoid = nn.Sigmoid()
        #self.act=SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)
        
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
        #self.act=SiLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

# ------------------------------ ViT start ----------------------------#
class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
 
        self.ln1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4*c, bias=False)
        self.fc2 = nn.Linear(4*c, c, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)
 
    def forward(self, x):
        x_ = self.ln1(x)
        x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_))[0]) + x
        x_ = self.ln2(x)
        x_ = self.fc2(self.dropout(self.act(self.fc1(x_))))
        x = x + self.dropout(x_)
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)

# ------------------------------ ViT End ----------------------------#

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

# ------------------------------ Swin Start --------------------------------------- #
'''
if yolov5l-xs-tph, then window_size = 8 and input_size is...
1) [B, 64, 160, 160]
2) [B, 128, 80, 80]
3) [B, 256, 40, 40]
4) [B, 512, 20, 20]

and C3(C)STR의 param은 앞단 C3에서의 hidden layer channel.
'''
def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerLayer(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c, num_heads, window_size=7, shift_size=0, 
                mlp_ratio = 4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if num_heads > 10:
            drop_path = 0.1
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(c)
        self.attn = WindowAttention(
            c, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(c)
        mlp_hidden_dim = int(c * mlp_ratio)
        self.mlp = Mlp(in_features=c, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = ( (0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, torch.tensor(-100.0)).masked_fill(attn_mask == 0, torch.tensor(0.0))
        return attn_mask

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.permute(0, 3, 2, 1).contiguous() # [b,h,w,c]

        attn_mask = self.create_mask(x, h, w) # [nW, Mh*Mw, Mh*Mw]
        shortcut = x
        x = self.norm1(x)
        
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            # print(f"shift size: {self.shift_size}")
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = window_partition(shifted_x, self.window_size) # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c) # [nW*B, Mh*Mw, C]

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # [B, H', W', C]
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x = x.permute(0, 3, 2, 1).contiguous()
        return x # (b, self.c2, w, h)

class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.tr = nn.Sequential(*(SwinTransformerLayer(c2, num_heads=num_heads, window_size=window_size,  shift_size=0 if (i % 2 == 0) else self.shift_size ) for i in range(num_layers)))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)

        x = self.tr(x)
        return x

# ------------------------------ Swin Over --------------------------------------- #

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)

class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_//32, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class ASPP(nn.Module):
    # Atrous Spatial Pyramid Pooling (ASPP) layer 
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.m = nn.ModuleList([nn.Conv2d(c_, c_, kernel_size=3, stride=1, padding=(x-1)//2, dilation=(x-1)//2, bias=False) for x in k])
        self.cv2 = Conv(c_ * (len(k) + 2), c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x]+ [self.maxpool(x)] + [m(x) for m in self.m] , 1))  

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model.model[-1]  # Detect()
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

''' mine '''
###################### CSwin Transformer #########################
# https://github.com/microsoft/CSWin-Transformer/blob/main/models/cswin.py
# MLP : Same with Swin

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    # print('img2windows 0) ', img.shape, H_sp, W_sp)
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    # print('img2windows 1) ', img_reshape.shape, ' because Hsp and Wsp is ', H_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous()
    # print('img2windows 2) ', img_reshape.shape)
    img_perm = img_perm.reshape(-1, H_sp * W_sp, C)
    # print('img2windows 3) ', img_perm.shape)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

# Role : WindowAttentinon in SWin
class LePEAttention(nn.Module):
    def __init__(self, dim, #resolution,
                 idx, split_size=7, dim_out=None, num_heads=8,
                 attn_drop=0., proj_drop=0., qk_scale=None):
        '''
        original:
        x.shape = [896, 32, 56, 1]
        H = 56
        W = 56
        H_sp = 56 = self.resolution
        W_sp = 1 = self.split_size
        '''
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        # self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.idx = idx
        # if idx == -1:
        #     H_sp, W_sp = self.resolution, self.resolution
        # elif idx == 0:
        #     H_sp, W_sp = self.resolution, self.split_size
        # elif idx == 1:
        #     W_sp, H_sp = self.resolution, self.split_size
        # else:
        #     print("ERROR MODE", idx)
        #     exit(0)
        # self.H_sp = H_sp
        # self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, Hsp, Wsp, _H, _W):
        '''
        x.shape = [1, 64, 256] --> H, W = 8(sqrt(64))

        '''
        B, N, C = x.shape
        H = _H
        W = _W
        # print('H, W : ', H, W)
        x = x.transpose(-2, -1).contiguous()
        # print('im2cswin 1 : ', x.shape)
        x = x.view(B, C, H, W)
        # print(x.shape, Hsp, Wsp)
        # print('im2cswin 2 : ', x.shape)
        x = img2windows(x, Hsp, Wsp)
        # print('im2cswin 3 : ', x.shape, ' because HSP and WSP is ', Hsp, Wsp)
        x = x.reshape(-1, Hsp * Wsp, self.num_heads, C // self.num_heads)
        # print('im2cswin 4 : ', x.shape, ' because headnum and C is ', self.num_heads, C)
        x = x.permute(0, 2, 1, 3).contiguous()
        # print('im2cswin 5 : ', x.shape)
        return x

    def get_lepe(self, x, func, Hsp, Wsp, _H, _W):
        B, N, C = x.shape
        H = _H
        W = _W
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = Hsp, Wsp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, Hsp * Wsp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, _H, _W):
        """
        x shape : [1, 64, 256] : B L C
        HW : HxW
        reso : sqrt(L)
        """
        if self.idx == -1:
            H_sp, W_sp = _H, _W
        elif self.idx == 0:
            H_sp, W_sp = _H, self.split_size
        elif self.idx == 1:
            H_sp, W_sp = self.split_size, _W
        else:
            print("ERROR MODE", self.idx)
            exit(0)

        # print('-------------------------------- 4) LePE Attn -------------------------------------- ')
        # print('qkv shape : ', qkv.shape)

        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        B, L, C = q.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        # print('var H W q shape : ', _H, _W, q.shape)
        # print('var Hsp Wsp : ', H_sp, W_sp)
        q = self.im2cswin(q, H_sp, W_sp, _H, _W)
        # print()
        k = self.im2cswin(k, H_sp, W_sp, _H, _W)
        # print('so, result of im2cswin : ', q.shape, k.shape)
        v, lepe = self.get_lepe(v, self.get_v, H_sp, W_sp, _H, _W)
        # print('so, result of get_lepe : ', v.shape, lepe.shape, ' becuase get_v is conv2 ', self.dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, H_sp, W_sp, _H, _W).view(B, -1, C)  # B H' W' C

        # print('lepe output shape ----- ', x.shape)
        return x

# same role with SwinTransformerLayer, original name is CSwinBlock
class CSwinBlock(nn.Module):
    '''
    original reso : img_size // (2 ** (i+1)), so If reso == split_size == 7, it is last stage.
    Here : input feature height(=width)
    deleted variance : about reso(=patches_resolution)
    '''
    def __init__(self, dim, num_heads, # reso
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        print(self.norm1)

        # self.patches_resolution, split_size = 112, 3 --> then, last_stage = False
        # if self.patches_resolution == split_size:
        #     last_stage = True
        # if last_stage:
        #     self.branch_num = 1
        # else:
        #     self.branch_num = 2
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        # if last_stage:
        #     self.attns = nn.ModuleList([
        #         LePEAttention(
        #             dim, idx = -1, # resolution=self.patches_resolution,
        #             split_size=split_size, num_heads=num_heads, dim_out=dim,
        #             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        #         for i in range(self.branch_num)])
        # else:
        #     self.attns = nn.ModuleList([
        #         LePEAttention(
        #             dim // 2, dix = i, # resolution=self.patches_resolution
        #             split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
        #             # dim, resolution=self.patches_resolution, idx=i,
        #             # split_size=split_size, num_heads=num_heads // 2, dim_out=dim,
        #             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        #         for i in range(self.branch_num)])

        self.attns = nn.ModuleList([
            LePEAttention(
                dim // 2, idx = i,  # resolution=self.patches_resolution
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                # dim, resolution=self.patches_resolution, idx=i,
                # split_size=split_size, num_heads=num_heads // 2, dim_out=dim,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        original : [16, 56 * 56, 64]
        cur : [1, 512, 8, 8] --> [1, 64, 512]
        """

        # H = W = self.patches_resolution
        # B, L, C = x.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        # input x.shape : 1, 512, 8, 8
        # print('------------------------- 3) CSwinBlock ------------------------------- ')
        # print('CSwin  input shape-------------', x.shape)

        # padding (also in SwinT)
        Padding = False
        _, _, H_, W_ = x.shape
        if min(H_, W_) < self.split_size or H_ % self.split_size!=0 or W_ % self.split_size!=0:
            # print('padding condition : ', H_, W_, self.split_size)
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) split_size {self.split_size}, Padding.')
            pad_r = (self.split_size - W_ % self.split_size) % self.split_size
            pad_b = (self.split_size - H_ % self.split_size) % self.split_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        # print('X after padding : ', x.shape)
        # padding over

        B, C, H, W = x.shape

        # 아래에서 2462줄 중 하나가 문제다. 무조건 output shape가 swint와 같도록 만들어라. swinT의 다음 인풋은역시 (1,512,8,8)이다.
        # 즉, [1,64,512]를 바꿔라.
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # b, L, c
        # print('after permute(0,2,3,1) and view(BLC) ', x.shape)

        img = self.norm1(x)
        # print('after norm : ', x.shape)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)    # 3, 1, 64, 512
        # print('qkv shape : ', qkv.shape)

        # branch로 나누는 이유 : 앞 절반은 horizontal, 뒤 절반은 vertical
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2], H, W)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], H, W)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # ----------------------------------- original CSwinT is over -------------------------------- #

        # change shape into 4 size, [batch, embed, height, width]
        x = x.permute(0, 2, 1).contiguous()
        # print('x shape after permute : ', x.shape)
        x = x.view(-1, C, H, W)  # b c h w

        # reverse padding
        if Padding:
            x = x[:, :, :H_, :W_]
        # print('cswin output shape : ', x.shape)
        # x = torch.mean(x, dim=1)    # added in 7/11 to match the dimension
        return x

# CSwin Transformer
# adapted only stage1
class CSwinTransformerBlock(nn.Module):
    '''
    num_heads == dim // 32 = 2/4/8/16
    yolov5l-tph-plus.yaml
        1) [B, 512, 20, 20]
    yolov5l-xs-tph:
        1) [B, 64, 160, 160]
        2) [B, 128, 80, 80]
        3) [B, 256, 40, 40]
        4) [B, 512, 20, 20]
    img_size는 forward에서 받게 한다.
    '''
    def __init__(self, c1, c2, num_heads, num_layers=1, img_size=20, split_size = [4,4,4,4]):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # print('CSwin init c1, c2, headnum, layernum, splitsize : ', c1, c2, num_heads, num_layers, split_size[0])

        # remove input_resolution
        self.blocks = nn.Sequential(*[CSwinBlock(dim=c2, num_heads=num_heads, # reso = img_size,
                                                      split_size = split_size[i]) for i in range(num_layers)])

    def forward(self, x):
        # print('---------------------- 1 + 2) CSWin + BasicLayer ------------------------ ')
        # print('SwinT input : ', x.shape)
        if self.conv is not None:
            x = self.conv(x)

        # print('SwinT after conv : ', x.shape)
        reso = x.shape[2]   # resolution
        x = self.blocks(x)
        # print('CSwinT output: ', x.shape)
        # print('----------------------------------------')
        return x

class C3CSTR(C3):
    # C3 module with CSwinTransformerBlock()
    # c_//32 = 16
    def __init__(self, c1, c2, num_layers=1, shortcut=True, g=1, e=0.5):
        # pdb.set_trace()
        super().__init__(c1, c2, num_layers, shortcut, g, e)
        c_ = int(c2 * e)    # 512
        self.m = CSwinTransformerBlock(c_, c_, c_//32, num_layers)

# class SWTCSPB(nn.Module):
#     # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(STCSPB, self).__init__()
#         c_ = int(c2)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1, 1)
#         num_heads = c_ // 32
#         self.m = CSwinTransformerBlock(c_, c_, num_heads, n)
#         #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         y1 = self.m(x1)
#         y2 = self.cv2(x1)
#         return self.cv3(torch.cat((y1, y2), dim=1))
#     pass

###################### CSwin Transformer End ######################

###################### HALO Net ############################

# relative positional embedding
def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class HaloAttention(nn.Module):
    '''
    original : __init__(self, *, dim, block_size, halo_size, dim_head=64, heads=8)
    c2 == dim
    num_heads == heads = c2//32(512일 때 16)
    block_size / halo_size : 8 / 4
    '''
    def __init__(self, c1, c2, num_heads, block_size, halo_size):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.heads = num_heads
        self.dim = c2
        dim_head = c2 // num_heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * num_heads
        print('inner_dim ', inner_dim)

        self.rel_pos_emb = RelPosEmb(
            block_size = block_size,
            rel_size = block_size + (halo_size * 2),
            dim_head = dim_head
        )

        self.to_q  = nn.Linear(c2, inner_dim, bias = False)
        self.to_kv = nn.Linear(c2, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, c2)

    def forward(self, x):
        '''
        yolov5l-tph-plus.yaml
            1) [B, 512, 20, 20] - train
            2) [B, 512, 12, 21] - val
        yolov5l-xs-tph:
            1) [B, 64, 160, 160] - train
            2) [B, 128, 80, 80]
            3) [B, 256, 40, 40]
            4) [B, 512, 20, 20]
        --> 모든 block/halo를 4/2로 통일, 이후에 차원 별로 각각 8/4, 16/8, 32/16으로 늘려보는 건 어떤가?
        '''
        b_, c_, h_, w_, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device

        # padding (also in SwinT)
        Padding = False
        if min(h_, w_) < self.block_size or h_ % self.block_size != 0 or w_ % self.block_size != 0:
            # print('padding condition : ', H_, W_, self.split_size)
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) split_size {self.split_size}, Padding.')
            pad_r = (self.block_size - w_ % self.block_size) % self.block_size
            pad_b = (self.block_size - h_ % self.block_size) % self.block_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        # print('X after padding : ', x.shape)
        # padding over

        b, c, h, w = x.shape

        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # get block neighborhoods, and prepare a halo-ed version (blocks with padding) for deriving key values
        # q_inp : [16, 64, 512] by block=8일 때, [32, 32]는 각각 4개씩 8x8로 나뉘어지니 [1x4x4, 8x8, 512]
        # print('x shape ', x.shape)
        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = block, p2 = block)
        # print('q_inp shape, p1, p2 : ', q_inp.shape, block, block)

        # x = [1, 512, 32, 32]일 떄, padding=4, stride=8, kernel=16이면
        # 0~16/8~24/16~32/24~40으로 가로 4회, 세로 4회 진행됨.
        # 또한, 이 16x16이 각각 512차원이므로, 총 [1, 512x256, 16] = [1, 131072, 16]가 됨.
        # 이를, rearrange 통해 batch와 패치를 묶고, 패치당 256개 픽셀, 각 픽셀은 512차원 임베딩 -> [1x16, 256, 512]
        kv_inp = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo)
        # print('kv unfold shape ', kv_inp.shape, 'kernelsize ', block+halo*2, 'block halo ', block, halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = c)
        # print('kv rearr shape ', kv_inp.shape)

        # derive queries, keys, values, here, inner_dim = 256
        # so q = [16, 64, 256] / k, v = [16, 256, 256]
        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim = -1)
        # print('q k v shape ', q.shape, k.shape, v.shape)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))
        # print('qkv after map ', q.shape, k.shape, v.shape)

        # scale
        q *= self.scale

        # attention, [64, 64, 256]
        sim = einsum('b i d, b j d -> b i j', q, k)
        # print('qkt(=sim) shape ', sim.shape)

        # add relative positional bias, still [64, 64, 256]
        sim += self.rel_pos_emb(q)
        # print('sim after pos_emb ', sim.shape)

        # mask out padding (in the paper, they claim to not need masks, but what about padding?)

        mask = torch.ones(1, 1, h, w, device = device)
        mask = F.unfold(mask, kernel_size = block + (halo * 2), stride = block, padding = halo)
        # print('mask unfold and block halo ', mask.shape, block, halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b = b, h = heads)
        # print('mask repeat d heads ', mask.shape, b, heads)
        # print(mask)
        mask = mask.bool()
        # print(mask)

        # This line computes the maximum negative value representable by the data type of the sim tensor.
        # This value is often used in attention mechanisms to mask out certain positions by setting their scores
        # to a very negative value, ensuring they get a near-zero weight after applying the softmax function
        max_neg_value = -torch.finfo(sim.dtype).max

        # https://thought-process-ing.tistory.com/79
        # sim의 바꾸고자 하는 값(mask)를 max_neg_value로 변경
        sim.masked_fill_(mask, max_neg_value)

        # attention
        attn = sim.softmax(dim = -1)

        # aggregate
        out = einsum('b i j, b j d -> b i d', attn, v)
        # print('out first shape ', out.shape)

        # merge and combine heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = heads)
        out = self.to_out(out)
        # print('out after rearr and to_out ', out.shape)

        # merge blocks back to original feature map
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b = b, h = (h // block), w = (w // block), p1 = block, p2 = block)

        # reverse padding
        if Padding:
            out = out[:, :, :h_, :w_]
        # print('final output ', out.shape)
        return out

class HaloAttentionBlock(nn.Module):
    '''
    num_heads == dim // 32 = 2/4/8/16
    yolov5l-tph-plus.yaml
        1) [B, 512, 20, 20]
    yolov5l-xs-tph:
        1) [B, 64, 160, 160]
        2) [B, 128, 80, 80]
        3) [B, 256, 40, 40]
        4) [B, 512, 20, 20]
    img_size는 forward에서 받게 한다.
    '''
    def __init__(self, c1, c2, num_heads, block_size, halo_size, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # print('CSwin init c1, c2, headnum, layernum, splitsize : ', c1, c2, num_heads, num_layers, split_size[0])

        # remove input_resolution
        self.blocks = nn.Sequential(*[HaloAttention(c1=c1, c2=c2, num_heads=num_heads, block_size=block_size,
                                                    halo_size=halo_size) for i in range(num_layers)])

    def forward(self, x):
        # print('---------------------- 1 + 2) CSWin + BasicLayer ------------------------ ')
        # print('SwinT input : ', x.shape)
        if self.conv is not None:
            x = self.conv(x)

        # print('SwinT after conv : ', x.shape)
        reso = x.shape[2]   # resolution
        x = self.blocks(x)
        # print('CSwinT output: ', x.shape)
        # print('----------------------------------------')
        return x

class C3Halo(C3):
    # C3 module with CSwinTransformerBlock()
    # c_//32 = 16
    def __init__(self, c1, c2, num_layers=1, shortcut=True, block_size=8, halo_size=4, g=1, e=0.5):
        super().__init__(c1, c2, num_layers, shortcut, g, e)
        c_ = int(c2 * e)    # 512
        self.m = HaloAttentionBlock(c_, c_, c_//32, block_size, halo_size, num_layers)
###################### HALO Net End #########################



###################### My Own Transformer Start ####################
class myAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, norm_layer=nn.LayerNorm, bias=False,
                 drop_path=0., mlp_ratio=4., act_layer=nn.GELU, drop=0.):
        super().__init__()

        # init parameter
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        # kernel size and scale
        self.kernelX = (window_size, window_size * 3)
        self.kernelY = (window_size * 3, window_size)

        # head variance, head_dim split by 2 because x divided into x1 and x2
        head_dim = (dim/2) // num_heads
        self.scale = head_dim ** -0.5

        # linear to make QKV
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim*3, bias)
        self.get_v = nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2)

        # Position Embedding
        # self.rel_pos_emb = RelPosEmb(
        #     block_size=window_size,
        #     rel_size=window_size * 3,
        #     dim_head=head_dim
        # )

        # Function after calculate QKV
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def get_lepe(self, x, func, flag):
        '''
        input : [Bhw, H/hw * W/ws, c]
        input must be [B, C, H, W] at func
        1) after transpose & contiguous : [Bhw c H/hw W/ws]
        2) so, after func, x : [Bhw c H/hw W/ws]
        3) lepe must be same size with qktv : [Bhw, H/ws * W/ws,

        이를 chan을 head로 나누고 그 head를 배치로 넣어야 함. 또한 HW 한꺼번에 만들어야 함.
        '''
        # print('5) get_lepe input : ', x.shape)

        B_, N_, C_ = x.shape

        if flag == 'x_axis':
            x = x.transpose(-2, -1).contiguous().view(B_, C_, self.window_size, self.window_size * 3)
            lepe = x[:, :, :, self.window_size : 2 * self.window_size]
        else :
            x = x.transpose(-2, -1).contiguous().view(B_, C_, self.window_size * 3, self.window_size)
            lepe = x[:, :, self.window_size : 2 * self.window_size, :]
        lepe = func(lepe)
        # print('6) lepe and x after func : ', x.shape, lepe.shape)

        # y : height, x : width
        x, lepe = map(lambda t: rearrange(t, 'b (h c) y x -> (b h) (y x) c', h = self.num_heads), (x, lepe))
        # print('7) get_lepe output x lepe ', x.shape, lepe.shape)
        # lepe를 reshape 해야 함.
        return x, lepe

    def Attn(self, x, flag, H, W):
        '''
        Input x : [3, B, HW, C/2(=c)]
        1) q_, k_, v_ = [B, HW, c]
        2-1) q_ : [Bhw, H/ws * W/ws, c] such as [16, 4, 2]
        2-2) k_/v_ : [Bhw, H/ws * W/ws, c] (단, padding값도 H/ws, W/ws에 포함) such as [Bx4x4, 12(2x6), 2(chan)]
        3) q, k, v after map(lambda t ... ) : [Bhw * head, H/ws * W/ws, c]
        '''
        # print('3) Attn input - ', x.shape)
        B_, N_, C_ = x.shape[1], x.shape[2], x.shape[3]

        # Implement qkv, here, divide into self.dim//2 because original input x is divided into x1/x2
        q_, k_, v_ = x[0], x[1], x[2]
        q_ = rearrange(q_, 'b (h w) c -> b h w c', h=H, w=W)
        q_ = rearrange(q_, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)

        k_ = rearrange(k_, 'b (h w) c -> b h w c', h=H, w=W).contiguous().permute(0, 3, 1, 2)
        v_ = rearrange(v_, 'b (h w) c -> b h w c', h=H, w=W).contiguous().permute(0, 3, 1, 2)

        if flag == 'x_axis':
            # print('-----------x axis-------------')
            k_ = F.pad(k_, (self.window_size, self.window_size, 0, 0))
            k_ = F.unfold(k_, kernel_size = (self.window_size, self.window_size * 3), stride = self.window_size)
            v_ = F.pad(v_, (self.window_size, self.window_size, 0, 0))
            v_ = F.unfold(v_, kernel_size=(self.window_size, self.window_size * 3), stride=self.window_size)
        else :
            # print('----------y axis--------------')
            k_ = F.pad(k_, (0, 0, self.window_size, self.window_size))
            k_ = F.unfold(k_, kernel_size=(self.window_size * 3, self.window_size), stride=self.window_size)
            v_ = F.pad(v_, (0, 0, self.window_size, self.window_size))
            v_ = F.unfold(v_, kernel_size=(self.window_size * 3, self.window_size), stride=self.window_size)
        k_ = rearrange(k_, 'b (c j) i -> (b i) j c', c=self.dim // 2)
        v_ = rearrange(v_, 'b (c j) i -> (b i) j c', c=self.dim // 2)
        # print('q_ k_ v_ : ', q_.shape, k_.shape, v_.shape, self.num_heads)

        # Divide Embedding into Head
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.num_heads), (q_, k_))
        q *= self.scale
        # print('4) q k v_ - ', q.shape, k.shape, v_.shape)

        # v에 get_lepe 또는 q에 rel_pos_emb 더하기
        v, lepe = self.get_lepe(v_, self.get_v, flag)

        # Attn 구하기
        sim = einsum('b i d, b j d -> b i j', q, k)

        # ------------ Halo Attn Mask & Position Embedding Start -------- #
        # sim += self.rel_pos_emb(q)
        #
        # # mask out padding (in the paper, they claim to not need masks, but what about padding?)
        # device = x.device
        # mask = torch.ones(1, 1, H, W, device=device)
        # mask = F.unfold(mask, kernel_size=self.window_size * 3, stride=self.window_size, padding=self.window_size)
        # # print('mask unfold and block halo ', mask.shape, block, halo)
        # mask = repeat(mask, '() j i -> (b i h) () j', b=B_, h=self.num_heads)
        # mask = mask.bool()
        #
        # max_neg_value = -torch.finfo(sim.dtype).max
        #
        # # https://thought-process-ing.tistory.com/79
        # # sim의 바꾸고자 하는 값(mask)를 max_neg_value로 변경
        # sim.masked_fill_(mask, max_neg_value)
        # ------------ Halo Attn Mask & Position Embedding End----------- #

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        # print('8) qktv - ', out.shape)
        out = out + lepe

        # merge and combine heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)

        # merge blocks back to original feature map
        out = rearrange(out, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', b=B_, h=(H//self.window_size),
                        w=(W//self.window_size), p1=self.window_size, p2=self.window_size)
        out = out.reshape(B_, -1, C_)

        # print('9) Attn output : ', out.shape)
        # print('------------------------')
        return out


    def forward(self, x):
        '''
        input x : [B, C, H, W]
        1) permute and view : [B, HW, C] (=[B, -1, C] 이후 norm)
        2) qkv : [3, B, HW, C]
        2-1) Attn input : [3, B, HW, C//2]
        3) Attend_x : cat(x1, x2) : [B, HW, C]
        '''
        B_, C_, H_, W_ = x.shape
        assert H_ >= self.window_size, 'window should be less than feature map size'
        assert W_ >= self.window_size, 'window should be less than feature map size'

        # H, W,가 window_size의 배수가 아닐 경우, 패딩
        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
            # print('padding condition : ', H_, W_, self.split_size)
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) split_size {self.split_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        # print('X after padding : ', x.shape)

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous().view(B_, H * W, C)
        # print('1) x [B HW C] - ', x.shape)

        # 우선 qkv를 만든 다음, channel을 1/2로 나눠, 하나는 가로방향, 하나는 세로방향으로 진행해야 됨.
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B_, H * W, 3, C).permute(2, 0, 1, 3) # [3, 1, 64, 4]
        # print('2) qkv ', qkv.shape)

        x1 = self.Attn(qkv[:, :, :, :C//2], 'x_axis', H, W)   # x-axis such as (2, 6). [3, 1, 64, 2]
        x2 = self.Attn(qkv[:, :, :, C//2:], 'y_axis', H, W)   # y-axis such as (6, 2). [3, 1, 64, 2]

        attened_x = torch.cat([x1, x2], dim=2)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # change shape into 4 size, [batch, embed, height, width]
        x = x.permute(0, 2, 1).contiguous()

        # print('x shape after permute : ', x.shape)
        x = x.view(-1, C, H, W)  # b c h w

        # print(Padding)
        # print(x.shape)
        # reverse padding
        if Padding:
            x = x[:, :, :H_, :W_]

        # print('10) Final output : ', x.shape)
        return x

class myAttnBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, window_size, num_layers=1):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.blocks = nn.Sequential(*[myAttention(dim=c2, num_heads=num_heads,
                                                 window_size = window_size) for i in range(num_layers)])

    def forward(self, x):
        # print('---------------------- 1 + 2) Mine + BasicLayer ------------------------ ')
        # print('SwinT input : ', x.shape)
        if self.conv is not None:
            x = self.conv(x)

        x = self.blocks(x)
        # print('MyT output: ', x.shape)
        # print('----------------------------------------')
        return x
class C3Mine(C3):
    # C3 module with CSwinTransformerBlock()
    # c_//32 = 16
    def __init__(self, c1, c2, num_layers=1, shortcut=True, window_size=4, g=1, e=0.5):
        super().__init__(c1, c2, num_layers, shortcut, g, e)
        c_ = int(c2 * e)    # 512
        self.m = myAttnBlock(c_, c_, c_//32, window_size, num_layers)   # c1, c2, num_heads, window_size
###################### My Own Transformer End ######################

# ##################### Vanila ViT start #########################
#
# # classes
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#
#         self.norm = nn.LayerNorm(dim)
#
#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         x = self.norm(x)
#
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
#                 FeedForward(dim, mlp_dim, dropout = dropout)
#             ]))
#
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#
#         return self.norm(x)
#
# class ViT(nn.Module):
#     def __init__(self, dim, num_heads):
#         super().__init__()
#
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2  = patch_width),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Linear(dim, num_classes)
#
#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, h, w = x.shape
#
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)
#
#         x = self.transformer(x)
#
#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
#
#         x = self.to_latent(x)
#         return self.mlp_head(x)
#
# class ViTBlock(nn.Module):
#     '''
#     num_heads == dim // 32 = 2/4/8/16
#     yolov5l-tph-plus.yaml
#         1) [B, 512, 20, 20]
#     yolov5l-xs-tph:
#         1) [B, 64, 160, 160]
#         2) [B, 128, 80, 80]
#         3) [B, 256, 40, 40]
#         4) [B, 512, 20, 20]
#     img_size는 forward에서 받게 한다.
#     '''
#     def __init__(self, c1, c2, num_heads, num_layers=1):
#         super().__init__()
#         self.conv = None
#         if c1 != c2:
#             self.conv = Conv(c1, c2)
#
#         # print('ViT init c1, c2, headnum, layernum, splitsize : ', c1, c2, num_heads, num_layers, split_size[0])
#
#         # remove input_resolution
#         self.blocks = nn.Sequential(*[ViT(dim=c2, num_heads=num_heads) for i in range(num_layers)])
#
#     def forward(self, x):
#         # print('---------------------- 1 + 2) CSWin + BasicLayer ------------------------ ')
#         # print('SwinT input : ', x.shape)
#         if self.conv is not None:
#             x = self.conv(x)
#
#         # print('SwinT after conv : ', x.shape)
#         x = self.blocks(x)
#         # print('CSwinT output: ', x.shape)
#         # print('----------------------------------------')
#         return x
#
# class C3ViT(C3):
#     # C3 module with CSwinTransformerBlock()
#     # c_//32 = 16
#     def __init__(self, c1, c2, num_layers=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, num_layers, shortcut, g, e)
#         c_ = int(c2 * e)    # 512
#         self.m = ViTBlock(c_, c_, c_//32, num_layers)   # c1, c2, num_heads, window_size
# ##################### Vanila ViT End ###########################
''' mine over '''