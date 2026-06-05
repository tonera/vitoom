import torch
from torch import nn
import torch.nn.functional as F


@torch.compile
class Silu(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)