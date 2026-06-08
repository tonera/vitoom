from __future__ import annotations

# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team.
# Licensed under the Apache License, Version 2.0
#
# This file is vendored and trimmed for inference runtime.

import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logging_utils import setup_logging
from ..safetensors_utils import load_safetensors_state_dict

setup_logging()
import logging

logger = logging.getLogger(__name__)

CACHE_T = 2
SCALE_FACTOR = 8


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None and generator.device.type != self.parameters.device.type:
            rand_device = generator.device
        else:
            rand_device = self.parameters.device
        sample = torch.randn(self.mean.shape, generator=generator, device=rand_device, dtype=self.parameters.dtype).to(self.parameters.device)
        return self.mean + self.std * sample

    def mode(self) -> torch.Tensor:
        return self.mean


class ChunkedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.spatial_chunk_size = kwargs.pop("spatial_chunk_size", None)
        super().__init__(*args, **kwargs)
        assert self.padding_mode == "zeros"
        assert self.dilation == (1, 1)
        assert self.groups == 1
        assert self.kernel_size[0] == self.kernel_size[1]
        assert self.stride[0] == self.stride[1]
        self.original_padding = self.padding
        self.padding = (0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_chunk_size is None or x.shape[2] <= self.spatial_chunk_size + self.kernel_size[0] + self.spatial_chunk_size // 4:
            self.padding = self.original_padding
            y = super().forward(x)
            self.padding = (0, 0)
            return y

        org_shape = x.shape
        overlap = self.kernel_size[0] // 2
        if self.original_padding[0] == 0:
            overlap = 0

        y_height = org_shape[2] // self.stride[0]
        y_width = org_shape[3] // self.stride[1]
        y = torch.zeros((org_shape[0], self.out_channels, y_height, y_width), dtype=x.dtype, device=x.device)
        yi = 0
        i = 0
        while i < org_shape[2]:
            si = i if i == 0 else i - overlap
            ei = i + self.spatial_chunk_size + overlap + self.stride[0] - 1
            if ei > org_shape[2] or ei + self.spatial_chunk_size // 4 > org_shape[2]:
                ei = org_shape[2]
            chunk = x[:, :, si:ei, :]
            if i == 0 and overlap > 0:
                chunk = F.pad(chunk, (overlap, overlap, overlap, 0), mode="constant", value=0)
            elif ei == org_shape[2] and overlap > 0:
                chunk = F.pad(chunk, (overlap, overlap, 0, overlap), mode="constant", value=0)
            elif overlap > 0:
                chunk = F.pad(chunk, (overlap, overlap), mode="constant", value=0)
            chunk = super().forward(chunk)
            y[:, :, yi : yi + chunk.shape[2], :] = chunk
            yi += chunk.shape[2]
            if ei == org_shape[2]:
                break
            i += self.spatial_chunk_size
        return y


class QwenImageCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        spatial_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)
        self.spatial_chunk_size = spatial_chunk_size
        self._supports_spatial_chunking = (
            self.groups == 1 and self.dilation[1] == 1 and self.dilation[2] == 1 and self.stride[1] == 1 and self.stride[2] == 1
        )

    def _forward_chunked_height(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = self.spatial_chunk_size
        if chunk_size is None or chunk_size <= 0 or not self._supports_spatial_chunking:
            return super().forward(x)

        kernel_h = self.kernel_size[1]
        if kernel_h <= 1 or x.shape[3] <= chunk_size:
            return super().forward(x)

        receptive_h = kernel_h
        out_h = x.shape[3] - receptive_h + 1
        if out_h <= 0:
            return super().forward(x)

        y0 = 0
        out = None
        while y0 < out_h:
            y1 = min(y0 + chunk_size, out_h)
            in0 = y0
            in1 = y1 + receptive_h - 1
            out_chunk = super().forward(x[:, :, :, in0:in1, :])
            if out is None:
                out_shape = list(out_chunk.shape)
                out_shape[3] = out_h
                out = out_chunk.new_empty(out_shape)
            out[:, :, :, y0:y1, :] = out_chunk
            y0 = y1
        return out

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self._forward_chunked_height(x)


class QwenImageRMS_norm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class QwenImageUpsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), ChunkedConv2d(dim, dim // 2, 3, padding=1))
        elif mode == "upsample3d":
            self.resample = nn.Sequential(QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), ChunkedConv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), ChunkedConv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), ChunkedConv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class QwenImageResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, non_linearity: str = "silu") -> None:
        assert non_linearity in ["silu"]
        super().__init__()
        self.nonlinearity = nn.SiLU()
        self.norm1 = QwenImageRMS_norm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImageCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)
        x = self.nonlinearity(self.norm1(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        x = self.dropout(self.nonlinearity(self.norm2(x)))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)
        return x + h


class QwenImageAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(batch_size * time, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)
        x = x.view(batch_size, time, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity


class QwenImageMidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class QwenImageEncoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        input_channels: int = 3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.nonlinearity = nn.SiLU()
        dims = [dim * u for u in [1] + dim_mult]
        self.conv_in = QwenImageCausalConv3d(input_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        scale = 1.0
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0
        self.mid_block = QwenImageMidBlock(out_dim, dropout, non_linearity, num_layers=1)
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
        x = self.mid_block(x, feat_cache, feat_idx)
        x = self.nonlinearity(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, dropout: float = 0.0, upsample_mode: Optional[str] = None, non_linearity: str = "silu"):
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)]) if upsample_mode is not None else None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx) if feat_cache is not None else resnet(x)
        if self.upsamplers is not None:
            up = self.upsamplers[0]
            x = up(x, feat_cache, feat_idx) if feat_cache is not None else up(x)
        return x


class QwenImageDecoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        output_channels: int = 3,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.nonlinearity = nn.SiLU()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dims[0], dropout, non_linearity, num_layers=1)
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
            self.up_blocks.append(QwenImageUpBlock(in_dim=in_dim, out_dim=out_dim, num_res_blocks=num_res_blocks, dropout=dropout, upsample_mode=upsample_mode))
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, output_channels, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        x = self.mid_block(x, feat_cache, feat_idx)
        for up in self.up_blocks:
            x = up(x, feat_cache, feat_idx) if feat_cache is not None else up(x)
        x = self.nonlinearity(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class AutoencoderKLQwenImage(nn.Module):
    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = None,
        latents_std: List[float] = None,
        input_channels: int = 3,
        spatial_chunk_size: Optional[int] = None,
        disable_cache: bool = False,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        self.encoder = QwenImageEncoder3d(base_dim, z_dim * 2, list(dim_mult), num_res_blocks, attn_scales, temperal_downsample, dropout, input_channels)
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder3d(base_dim, z_dim, list(dim_mult), num_res_blocks, attn_scales, self.temperal_upsample, dropout, input_channels)
        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        self.spatial_chunk_size = None
        if spatial_chunk_size is not None and spatial_chunk_size > 0:
            self.enable_spatial_chunking(spatial_chunk_size)

        self.cache_disabled = False
        if disable_cache:
            self.disable_cache()

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def enable_spatial_chunking(self, spatial_chunk_size: int) -> None:
        if spatial_chunk_size is None or spatial_chunk_size <= 0:
            raise ValueError
        self.spatial_chunk_size = int(spatial_chunk_size)
        for module in self.modules():
            if isinstance(module, QwenImageCausalConv3d):
                module.spatial_chunk_size = self.spatial_chunk_size
            elif isinstance(module, ChunkedConv2d):
                module.spatial_chunk_size = self.spatial_chunk_size

    def disable_cache(self) -> None:
        self.cache_disabled = True
        self.clear_cache = lambda: None
        self._feat_map = None
        self._enc_feat_map = None

    def clear_cache(self):
        def _count_conv3d(model):
            return sum(isinstance(m, QwenImageCausalConv3d) for m in model.modules())

        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, _, _ = x.shape
        assert num_frame == 1 or not self.cache_disabled
        self.clear_cache()
        iter_ = 1 + (num_frame - 1) // 4
        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return {"latent_dist": posterior}

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, _, _ = z.shape
        assert num_frame == 1 or not self.cache_disabled
        self.clear_cache()
        x = self.post_quant_conv(z)
        out = None
        for i in range(num_frame):
            self._conv_idx = [0]
            out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = out_ if out is None else torch.cat([out, out_], 2)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)
        return {"sample": out}

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice)["sample"] for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)["sample"]
        if not return_dict:
            return (decoded,)
        return {"sample": decoded}

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        is_4d = latents.dim() == 4
        if is_4d:
            latents = latents.unsqueeze(2)
        latents = latents.to(self.dtype)
        latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        image = self.decode(latents, return_dict=False)[0]
        if is_4d:
            image = image.squeeze(2)
        return image.clamp(-1.0, 1.0)

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        is_4d = pixels.dim() == 4
        if is_4d:
            pixels = pixels.unsqueeze(2)
        pixels = pixels.to(self.dtype)
        posterior = self.encode(pixels, return_dict=False)[0]
        latents = posterior.mode()
        latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std
        if is_4d:
            latents = latents.squeeze(2)
        return latents


def convert_comfyui_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Heuristic: official format doesn't have conv1.bias at top-level.
    if "conv1.bias" not in sd:
        return sd
    # Full key mapping ported from sd-scripts for compatibility.
    key_map = {
        "conv1": "quant_conv",
        "conv2": "post_quant_conv",
        "decoder.conv1": "decoder.conv_in",
        "decoder.head.0": "decoder.norm_out",
        "decoder.head.2": "decoder.conv_out",
        "decoder.middle.0.residual.0": "decoder.mid_block.resnets.0.norm1",
        "decoder.middle.0.residual.2": "decoder.mid_block.resnets.0.conv1",
        "decoder.middle.0.residual.3": "decoder.mid_block.resnets.0.norm2",
        "decoder.middle.0.residual.6": "decoder.mid_block.resnets.0.conv2",
        "decoder.middle.1.norm": "decoder.mid_block.attentions.0.norm",
        "decoder.middle.1.proj": "decoder.mid_block.attentions.0.proj",
        "decoder.middle.1.to_qkv": "decoder.mid_block.attentions.0.to_qkv",
        "decoder.middle.2.residual.0": "decoder.mid_block.resnets.1.norm1",
        "decoder.middle.2.residual.2": "decoder.mid_block.resnets.1.conv1",
        "decoder.middle.2.residual.3": "decoder.mid_block.resnets.1.norm2",
        "decoder.middle.2.residual.6": "decoder.mid_block.resnets.1.conv2",
        "decoder.upsamples.0.residual.0": "decoder.up_blocks.0.resnets.0.norm1",
        "decoder.upsamples.0.residual.2": "decoder.up_blocks.0.resnets.0.conv1",
        "decoder.upsamples.0.residual.3": "decoder.up_blocks.0.resnets.0.norm2",
        "decoder.upsamples.0.residual.6": "decoder.up_blocks.0.resnets.0.conv2",
        "decoder.upsamples.1.residual.0": "decoder.up_blocks.0.resnets.1.norm1",
        "decoder.upsamples.1.residual.2": "decoder.up_blocks.0.resnets.1.conv1",
        "decoder.upsamples.1.residual.3": "decoder.up_blocks.0.resnets.1.norm2",
        "decoder.upsamples.1.residual.6": "decoder.up_blocks.0.resnets.1.conv2",
        "decoder.upsamples.10.residual.0": "decoder.up_blocks.2.resnets.2.norm1",
        "decoder.upsamples.10.residual.2": "decoder.up_blocks.2.resnets.2.conv1",
        "decoder.upsamples.10.residual.3": "decoder.up_blocks.2.resnets.2.norm2",
        "decoder.upsamples.10.residual.6": "decoder.up_blocks.2.resnets.2.conv2",
        "decoder.upsamples.11.resample.1": "decoder.up_blocks.2.upsamplers.0.resample.1",
        "decoder.upsamples.12.residual.0": "decoder.up_blocks.3.resnets.0.norm1",
        "decoder.upsamples.12.residual.2": "decoder.up_blocks.3.resnets.0.conv1",
        "decoder.upsamples.12.residual.3": "decoder.up_blocks.3.resnets.0.norm2",
        "decoder.upsamples.12.residual.6": "decoder.up_blocks.3.resnets.0.conv2",
        "decoder.upsamples.13.residual.0": "decoder.up_blocks.3.resnets.1.norm1",
        "decoder.upsamples.13.residual.2": "decoder.up_blocks.3.resnets.1.conv1",
        "decoder.upsamples.13.residual.3": "decoder.up_blocks.3.resnets.1.norm2",
        "decoder.upsamples.13.residual.6": "decoder.up_blocks.3.resnets.1.conv2",
        "decoder.upsamples.14.residual.0": "decoder.up_blocks.3.resnets.2.norm1",
        "decoder.upsamples.14.residual.2": "decoder.up_blocks.3.resnets.2.conv1",
        "decoder.upsamples.14.residual.3": "decoder.up_blocks.3.resnets.2.norm2",
        "decoder.upsamples.14.residual.6": "decoder.up_blocks.3.resnets.2.conv2",
        "decoder.upsamples.2.residual.0": "decoder.up_blocks.0.resnets.2.norm1",
        "decoder.upsamples.2.residual.2": "decoder.up_blocks.0.resnets.2.conv1",
        "decoder.upsamples.2.residual.3": "decoder.up_blocks.0.resnets.2.norm2",
        "decoder.upsamples.2.residual.6": "decoder.up_blocks.0.resnets.2.conv2",
        "decoder.upsamples.3.resample.1": "decoder.up_blocks.0.upsamplers.0.resample.1",
        "decoder.upsamples.3.time_conv": "decoder.up_blocks.0.upsamplers.0.time_conv",
        "decoder.upsamples.4.residual.0": "decoder.up_blocks.1.resnets.0.norm1",
        "decoder.upsamples.4.residual.2": "decoder.up_blocks.1.resnets.0.conv1",
        "decoder.upsamples.4.residual.3": "decoder.up_blocks.1.resnets.0.norm2",
        "decoder.upsamples.4.residual.6": "decoder.up_blocks.1.resnets.0.conv2",
        "decoder.upsamples.4.shortcut": "decoder.up_blocks.1.resnets.0.conv_shortcut",
        "decoder.upsamples.5.residual.0": "decoder.up_blocks.1.resnets.1.norm1",
        "decoder.upsamples.5.residual.2": "decoder.up_blocks.1.resnets.1.conv1",
        "decoder.upsamples.5.residual.3": "decoder.up_blocks.1.resnets.1.norm2",
        "decoder.upsamples.5.residual.6": "decoder.up_blocks.1.resnets.1.conv2",
        "decoder.upsamples.6.residual.0": "decoder.up_blocks.1.resnets.2.norm1",
        "decoder.upsamples.6.residual.2": "decoder.up_blocks.1.resnets.2.conv1",
        "decoder.upsamples.6.residual.3": "decoder.up_blocks.1.resnets.2.norm2",
        "decoder.upsamples.6.residual.6": "decoder.up_blocks.1.resnets.2.conv2",
        "decoder.upsamples.7.resample.1": "decoder.up_blocks.1.upsamplers.0.resample.1",
        "decoder.upsamples.7.time_conv": "decoder.up_blocks.1.upsamplers.0.time_conv",
        "decoder.upsamples.8.residual.0": "decoder.up_blocks.2.resnets.0.norm1",
        "decoder.upsamples.8.residual.2": "decoder.up_blocks.2.resnets.0.conv1",
        "decoder.upsamples.8.residual.3": "decoder.up_blocks.2.resnets.0.norm2",
        "decoder.upsamples.8.residual.6": "decoder.up_blocks.2.resnets.0.conv2",
        "decoder.upsamples.9.residual.0": "decoder.up_blocks.2.resnets.1.norm1",
        "decoder.upsamples.9.residual.2": "decoder.up_blocks.2.resnets.1.conv1",
        "decoder.upsamples.9.residual.3": "decoder.up_blocks.2.resnets.1.norm2",
        "decoder.upsamples.9.residual.6": "decoder.up_blocks.2.resnets.1.conv2",
        "encoder.conv1": "encoder.conv_in",
        "encoder.downsamples.0.residual.0": "encoder.down_blocks.0.norm1",
        "encoder.downsamples.0.residual.2": "encoder.down_blocks.0.conv1",
        "encoder.downsamples.0.residual.3": "encoder.down_blocks.0.norm2",
        "encoder.downsamples.0.residual.6": "encoder.down_blocks.0.conv2",
        "encoder.downsamples.1.residual.0": "encoder.down_blocks.1.norm1",
        "encoder.downsamples.1.residual.2": "encoder.down_blocks.1.conv1",
        "encoder.downsamples.1.residual.3": "encoder.down_blocks.1.norm2",
        "encoder.downsamples.1.residual.6": "encoder.down_blocks.1.conv2",
        "encoder.downsamples.10.residual.0": "encoder.down_blocks.10.norm1",
        "encoder.downsamples.10.residual.2": "encoder.down_blocks.10.conv1",
        "encoder.downsamples.10.residual.3": "encoder.down_blocks.10.norm2",
        "encoder.downsamples.10.residual.6": "encoder.down_blocks.10.conv2",
        "encoder.downsamples.2.resample.1": "encoder.down_blocks.2.resample.1",
        "encoder.downsamples.3.residual.0": "encoder.down_blocks.3.norm1",
        "encoder.downsamples.3.residual.2": "encoder.down_blocks.3.conv1",
        "encoder.downsamples.3.residual.3": "encoder.down_blocks.3.norm2",
        "encoder.downsamples.3.residual.6": "encoder.down_blocks.3.conv2",
        "encoder.downsamples.3.shortcut": "encoder.down_blocks.3.conv_shortcut",
        "encoder.downsamples.4.residual.0": "encoder.down_blocks.4.norm1",
        "encoder.downsamples.4.residual.2": "encoder.down_blocks.4.conv1",
        "encoder.downsamples.4.residual.3": "encoder.down_blocks.4.norm2",
        "encoder.downsamples.4.residual.6": "encoder.down_blocks.4.conv2",
        "encoder.downsamples.5.resample.1": "encoder.down_blocks.5.resample.1",
        "encoder.downsamples.5.time_conv": "encoder.down_blocks.5.time_conv",
        "encoder.downsamples.6.residual.0": "encoder.down_blocks.6.norm1",
        "encoder.downsamples.6.residual.2": "encoder.down_blocks.6.conv1",
        "encoder.downsamples.6.residual.3": "encoder.down_blocks.6.norm2",
        "encoder.downsamples.6.residual.6": "encoder.down_blocks.6.conv2",
        "encoder.downsamples.6.shortcut": "encoder.down_blocks.6.conv_shortcut",
        "encoder.downsamples.7.residual.0": "encoder.down_blocks.7.norm1",
        "encoder.downsamples.7.residual.2": "encoder.down_blocks.7.conv1",
        "encoder.downsamples.7.residual.3": "encoder.down_blocks.7.norm2",
        "encoder.downsamples.7.residual.6": "encoder.down_blocks.7.conv2",
        "encoder.downsamples.8.resample.1": "encoder.down_blocks.8.resample.1",
        "encoder.downsamples.8.time_conv": "encoder.down_blocks.8.time_conv",
        "encoder.downsamples.9.residual.0": "encoder.down_blocks.9.norm1",
        "encoder.downsamples.9.residual.2": "encoder.down_blocks.9.conv1",
        "encoder.downsamples.9.residual.3": "encoder.down_blocks.9.norm2",
        "encoder.downsamples.9.residual.6": "encoder.down_blocks.9.conv2",
        "encoder.head.0": "encoder.norm_out",
        "encoder.head.2": "encoder.conv_out",
        "encoder.middle.0.residual.0": "encoder.mid_block.resnets.0.norm1",
        "encoder.middle.0.residual.2": "encoder.mid_block.resnets.0.conv1",
        "encoder.middle.0.residual.3": "encoder.mid_block.resnets.0.norm2",
        "encoder.middle.0.residual.6": "encoder.mid_block.resnets.0.conv2",
        "encoder.middle.1.norm": "encoder.mid_block.attentions.0.norm",
        "encoder.middle.1.proj": "encoder.mid_block.attentions.0.proj",
        "encoder.middle.1.to_qkv": "encoder.mid_block.attentions.0.to_qkv",
        "encoder.middle.2.residual.0": "encoder.mid_block.resnets.1.norm1",
        "encoder.middle.2.residual.2": "encoder.mid_block.resnets.1.conv1",
        "encoder.middle.2.residual.3": "encoder.mid_block.resnets.1.norm2",
        "encoder.middle.2.residual.6": "encoder.mid_block.resnets.1.conv2",
    }
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        new_key = key
        key_without_suffix = key.rsplit(".", 1)[0]
        if key_without_suffix in key_map:
            new_key = key.replace(key_without_suffix, key_map[key_without_suffix])
        new_state_dict[new_key] = value
    logger.info("Converted ComfyUI VAE keys -> official format")
    return new_state_dict


def load_vae(
    vae_path: str,
    *,
    input_channels: int = 3,
    device: Union[str, torch.device] = "cpu",
    spatial_chunk_size: Optional[int] = None,
    disable_cache: bool = False,
) -> AutoencoderKLQwenImage:
    VAE_CONFIG_JSON = """
{
  "attn_scales": [],
  "base_dim": 96,
  "dim_mult": [1, 2, 4, 4],
  "dropout": 0.0,
  "latents_mean": [-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921],
  "latents_std": [2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.916],
  "num_res_blocks": 2,
  "temperal_downsample": [false, true, true],
  "z_dim": 16
}
"""
    if spatial_chunk_size is not None and spatial_chunk_size % 2 != 0:
        spatial_chunk_size += 1
        logger.warning(f"Adjusted spatial_chunk_size to even: {spatial_chunk_size}")

    cfg = json.loads(VAE_CONFIG_JSON)
    vae = AutoencoderKLQwenImage(
        base_dim=cfg["base_dim"],
        z_dim=cfg["z_dim"],
        dim_mult=tuple(cfg["dim_mult"]),
        num_res_blocks=cfg["num_res_blocks"],
        attn_scales=cfg["attn_scales"],
        temperal_downsample=cfg["temperal_downsample"],
        dropout=cfg["dropout"],
        latents_mean=cfg["latents_mean"],
        latents_std=cfg["latents_std"],
        input_channels=input_channels,
        spatial_chunk_size=spatial_chunk_size,
        disable_cache=disable_cache,
    )

    logger.info(f"Loading VAE weights from {vae_path}")
    sd = load_safetensors_state_dict(vae_path, device="cpu", dtype=None)
    sd = convert_comfyui_state_dict(sd)
    info = vae.load_state_dict(sd, strict=False)
    if info.unexpected_keys:
        raise RuntimeError(f"VAE unexpected keys: {info.unexpected_keys[:10]}")
    if info.missing_keys:
        # 宽松一些：不同权重/版本可能有少量 buffer 差异
        logger.warning(f"VAE missing keys: {info.missing_keys[:10]}")

    vae.to(device)
    return vae

