import mlx.nn as nn


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, vocab_size)

    def __call__(self, token_ids):
        logits = self.token_table(token_ids)
        return logits