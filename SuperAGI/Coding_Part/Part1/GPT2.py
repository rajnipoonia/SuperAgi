class GPT2(nn.Module):
    def __init__(self, embed_size, num_layers, heads, vocab_size, forward_expansion, dropout, max_length):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.embedding = EmbeddingLayer(vocab_size, embed_size, max_length)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        return out
