import torch
import math

def rotary_embedding(dim, max_len=512):
    # Assuming dim is even
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).type_as(inv_freq)
    freqs = t[:, None] * inv_freq[None]
    emb = torch.cat((freqs, freqs), dim=-1).view(max_len, -1)
    return torch.sin(emb), torch.cos(emb)

def apply_rotary_emb(qk, cos, sin):
    return (qk * cos) + torch.roll(qk, 1, dims=-1) * sin

class RotaryEmbeddingLayer(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.sin, self.cos = rotary_embedding(dim, max_len)

    def forward(self, qk):
        return apply_rotary_emb(qk, self.cos, self.sin)

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_size, heads, group_size):
        # Initialization code here
        pass

    def forward(self, queries, keys, values):
        # Split queries into groups
        # Apply attention within each group
        # Combine the results from each group
        return output
    
def sliding_window_mask(size, window_size, device="cpu"):
    center = size // 2
    mask = torch.abs(torch.arange(size) - center) <= window_size
    return mask.to(device)

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        # Initialization code here
        pass

    def forward(self, queries, keys, values):
        # Compute sliding window mask
        # Apply attention only within the window for each token
        return output
