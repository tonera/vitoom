import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (for multimodal RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    mrope_interleaved: bool = False,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies 3D multimodal RoPE (temporal, height, width) to q and k.

    cos, sin: (3, batch, seq_len, head_dim) from TalkerRotaryEmbedding.
    mrope_section: e.g. [24, 20, 20] for head_dim/2 split (24+20+20=64).
    """
    if mrope_interleaved:

        def apply_interleaved_rope(x: torch.Tensor, modality_num: int) -> torch.Tensor:
            x_t = x[0].clone()
            index_ranges = []
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                index_ranges.append((beg_idx, end_idx))
            for beg_idx, end_idx in index_ranges:
                x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(
            unsqueeze_dim
        )
    else:
        mrope_section = [s * 2 for s in mrope_section]
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 1D RoPE to query and key. Returns rotated q, k (same interface as TalkerRotaryEmbedding)."""
        # Normalize positions to (batch, seq_len)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        batch, seq_len = positions.shape[0], positions.shape[1]
        num_heads = query.shape[1]
        head_size = query.shape[2]

        # Reshape (B*L, H, D) -> (B, H, L, D) for apply_rotary_pos_emb
        query_4d = query.view(batch, seq_len, num_heads, head_size).transpose(1, 2)
        key_4d = key.view(batch, seq_len, key.shape[1], head_size).transpose(1, 2)

        with torch.no_grad():
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(batch, -1, 1)
                .to(query.device)
            )
            position_ids_expanded = positions[:, None, :].float()

            device_type = (
                query.device.type
                if isinstance(query.device.type, str) and query.device.type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos().to(query.dtype)
                sin = emb.sin().to(query.dtype)

        q_embed, k_embed = apply_rotary_pos_emb(
            query_4d, key_4d, cos, sin, unsqueeze_dim=1
        )
        # (B, H, L, D) -> (B*L, H, D)
        return (
            q_embed.transpose(1, 2).reshape(batch * seq_len, num_heads, head_size),
            k_embed.transpose(1, 2).reshape(batch * seq_len, key.shape[1], head_size),
        )

class TalkerRotaryEmbedding(nn.Module):
    """3D RoPE for Talker (temporal, height, width). Expects positions of shape (3, batch, seq_len)."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
        attention_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.attention_scaling = attention_scaling
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (3, batch, seq_len), (batch, seq_len), or (seq_len,); normalize to (3, batch, seq_len)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)  # (seq_len,) -> (3, 1, seq_len)
        elif positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(3, -1, -1)  # (batch, seq_len) -> (3, batch, seq_len)
        batch, seq_len = positions.shape[1], positions.shape[2]
        num_heads = query.shape[1]
        # Reshape (B*L, H, D) -> (B, H, L, D) for apply_multimodal (expects batch, heads, seq_len, head_dim)
        query_4d = query.view(batch, seq_len, num_heads, self.head_size).transpose(1, 2)
        key_4d = key.view(batch, seq_len, key.shape[1], self.head_size).transpose(1, 2)

        # Match Qwen3-TTS: compute RoPE in float32 (autocast disabled) then scale cos/sin
        inv_freq = self.inv_freq.to(query.device).float()
        position_ids_expanded = positions[:, :, None, :].float()  # (3, batch, 1, seq_len)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(
            3, batch, -1, 1
        ).to(positions.device)
        device_type = "cuda" if query.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = (emb.cos() * self.attention_scaling).to(query.dtype)
            sin = (emb.sin() * self.attention_scaling).to(query.dtype)

        q_embed, k_embed = apply_multimodal_rotary_pos_emb(
            query_4d, key_4d, cos, sin,
            self.mrope_section,
            self.mrope_interleaved,
            unsqueeze_dim=1,
        )
        # (B, H, L, D) -> (B*L, H, D)
        return (
            q_embed.transpose(1, 2).reshape(batch * seq_len, num_heads, self.head_size),
            k_embed.transpose(1, 2).reshape(batch * seq_len, key.shape[1], self.head_size),
        )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    if rope_scaling is not None and "mrope_section" in rope_scaling:
        return TalkerRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            mrope_section=rope_scaling["mrope_section"],
            mrope_interleaved=rope_scaling.get("interleaved", False),
            attention_scaling=rope_scaling.get("attention_scaling", 1.0),
        )
    # Standard 1D RoPE (matches Qwen3TTSRotaryEmbedding)
    return RotaryEmbedding(
        rotary_dim=rotary_dim   ,
        max_position_embeddings=max_position,
        base=base,
    )
