import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=512):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_len, embed_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        return self.token_embeddings(x) + self.position_embeddings(positions)
