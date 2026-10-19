import mlx.nn as nn
import mlx.core as mx

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, vocab_size)

    def __call__(self, token_ids):
        logits = self.token_table(token_ids)
        return logits

    def loss(self, token_ids, targets):
        logits = self(token_ids)
        batch_size, context_length, vocab_size = logits.shape
        # cross_entropy wants one row of scores per token, so flatten the batch
        logits = logits.reshape(batch_size * context_length, vocab_size)
        targets = targets.reshape(batch_size * context_length)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def generate(self, token_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(token_ids)
            last_logits = logits[:, -1, :]
            next_token_id = mx.random.categorical(last_logits)
            next_token_id = next_token_id.reshape(-1 , 1)
            token_ids = mx.concatenate([token_ids, next_token_id], axis=1)
        return token_ids