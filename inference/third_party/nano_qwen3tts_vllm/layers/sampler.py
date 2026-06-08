import torch
from torch import nn
import time

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        
    def apply_temperature(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return logits.float().div_(temperatures.unsqueeze(dim=1))
    

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_k: int=50, top_p: float=1.0):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        top_k_logits = logits.masked_fill(indices_to_remove, -float("Inf"))
        
        probs = torch.softmax(top_k_logits, dim=-1)
        sample_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return sample_tokens
