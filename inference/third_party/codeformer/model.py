"""
Vendored minimal CodeFormer network (inference only).

Upstream: https://github.com/sczhou/CodeFormer
License: S-Lab License 1.0 (see LICENSE)

Notes:
- This is a minimal, self-contained subset sufficient for loading `codeformer.pth`
  and running forward() for face restoration.
- Registry/logging utilities from upstream basicsr are intentionally removed.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, emb_dim: int, beta: float) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.emb_dim = int(emb_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z: Tensor):
        # BCHW -> BHWC
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.emb_dim)

        d = (
            (z_flat**2).sum(dim=1, keepdim=True)
            + (self.embedding.weight**2).sum(dim=1)
            - 2.0 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        min_idx = torch.argmin(d, dim=1).unsqueeze(1)

        enc = torch.zeros(min_idx.shape[0], self.codebook_size, device=z.device, dtype=z.dtype)
        enc.scatter_(1, min_idx, 1)

        z_q = torch.matmul(enc, self.embedding.weight).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()

        # back to BCHW
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, {"min_encoding_indices": min_idx}

    def get_codebook_feat(self, indices: Tensor, shape):
        # indices: (B, HW, 1) or (B*HW, 1) -> (B*HW, 1)
        indices = indices.view(-1, 1)
        enc = torch.zeros(indices.shape[0], self.codebook_size, device=indices.device, dtype=torch.float32)
        enc.scatter_(1, indices, 1.0)
        z_q = torch.matmul(enc, self.embedding.weight)  # (B*HW, C)
        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        return z_q


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels if out_channels is None else out_channels)
        self.norm1 = _normalize(self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _normalize(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = None
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in: Tensor) -> Tensor:
        x = self.norm1(x_in)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        x_skip = self.conv_out(x_in) if self.conv_out is not None else x_in
        return x + x_skip


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.norm = _normalize(self.in_channels)
        self.q = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        w_ = torch.bmm(q, k) * (float(c) ** -0.5)
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nf: int,
        emb_dim: int,
        ch_mult: List[int],
        num_res_blocks: int,
        resolution: int,
        attn_resolutions: List[int],
    ) -> None:
        super().__init__()
        self.resolution = int(resolution)
        self.attn_resolutions = list(attn_resolutions)
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = int(num_res_blocks)

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks: List[nn.Module] = []
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        blocks.append(_normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        nf: int,
        emb_dim: int,
        ch_mult: List[int],
        res_blocks: int,
        img_size: int,
        attn_resolutions: List[int],
    ) -> None:
        super().__init__()
        self.nf = int(nf)
        self.ch_mult = list(ch_mult)
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = int(res_blocks)
        self.resolution = int(img_size)
        self.attn_resolutions = list(attn_resolutions)

        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        blocks: List[nn.Module] = []
        blocks.append(nn.Conv2d(emb_dim, block_in_ch, kernel_size=3, stride=1, padding=1))

        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(_normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, 3, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VQAutoEncoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        nf: int,
        ch_mult: List[int],
        quantizer: str = "nearest",
        res_blocks: int = 2,
        attn_resolutions: List[int] = [16],
        codebook_size: int = 1024,
        emb_dim: int = 256,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = 3
        self.nf = int(nf)
        self.n_blocks = int(res_blocks)
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(emb_dim)
        self.ch_mult = list(ch_mult)
        self.resolution = int(img_size)
        self.attn_resolutions = list(attn_resolutions)
        self.quantizer_type = str(quantizer)

        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
        )

        if self.quantizer_type != "nearest":
            raise ValueError("Only 'nearest' quantizer is supported in this minimal build.")
        self.beta = float(beta)
        self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)

        self.generator = Generator(
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
        )

    def forward(self, x: Tensor):
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, quant_stats


def calc_mean_std(feat: Tensor, eps: float = 1e-5):
    size = feat.size()
    if len(size) != 4:
        raise ValueError("The input feature should be 4D tensor.")
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim: int, nhead: int = 8, dim_mlp: int = 2048, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(t: Tensor, pos: Optional[Tensor]):
        return t if pos is None else t + pos

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class FuseSFTBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, enc_feat: Tensor, dec_feat: Tensor, w: float = 1.0) -> Tensor:
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = float(w) * (dec_feat * scale + shift)
        return dec_feat + residual


class CodeFormer(VQAutoEncoder):
    def __init__(
        self,
        dim_embd: int = 512,
        n_head: int = 8,
        n_layers: int = 9,
        codebook_size: int = 1024,
        latent_size: int = 256,
        connect_list: List[str] = ["32", "64", "128", "256"],
        fix_modules: Optional[List[str]] = ["quantize", "generator"],
        vqgan_path: Optional[str] = None,
    ) -> None:
        super().__init__(img_size=512, nf=64, ch_mult=[1, 2, 2, 4, 4, 8], quantizer="nearest", res_blocks=2, attn_resolutions=[16], codebook_size=codebook_size, emb_dim=256)

        if vqgan_path is not None:
            sd = torch.load(vqgan_path, map_location="cpu")
            state = sd.get("params_ema") if isinstance(sd, dict) else sd
            if state is not None:
                self.load_state_dict(state, strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                m = getattr(self, module, None)
                if m is None:
                    continue
                for p in m.parameters():
                    p.requires_grad = False

        self.connect_list = list(connect_list)
        self.n_layers = int(n_layers)
        self.dim_embd = int(dim_embd)
        self.dim_mlp = int(dim_embd) * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        self.ft_layers = nn.Sequential(
            *[
                TransformerSALayer(embed_dim=self.dim_embd, nhead=int(n_head), dim_mlp=self.dim_mlp, dropout=0.0)
                for _ in range(self.n_layers)
            ]
        )

        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(self.dim_embd), nn.Linear(self.dim_embd, codebook_size, bias=False))

        self.channels = {"16": 512, "32": 256, "64": 256, "128": 128, "256": 128, "512": 64}
        self.fuse_encoder_block = {"512": 2, "256": 5, "128": 8, "64": 11, "32": 14, "16": 18}
        self.fuse_generator_block = {"16": 6, "32": 9, "64": 12, "128": 15, "256": 18, "512": 21}

        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = FuseSFTBlock(in_ch, in_ch)

        # match upstream init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: Tensor, w: float = 0.0, detach_16: bool = True, code_only: bool = False, adain: bool = False):
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        lq_feat = x

        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)  # (hw) b c
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))  # (hw) b c
        query_emb = feat_emb
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        logits = self.idx_pred_layer(query_emb).permute(1, 0, 2)  # b(hw)n
        if code_only:
            return logits, lq_feat

        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])
        if detach_16:
            quant_feat = quant_feat.detach()
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in fuse_list:
                f_size = str(x.shape[-1])
                if w and float(w) > 0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, float(w))
        out = x
        return out, logits, lq_feat

