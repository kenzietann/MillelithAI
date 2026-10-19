import mlx.nn as nn
import mlx.core as mx

from aether.attention import SelfAttentionHead

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


class AttentionModel(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_size, head_size):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_length, embedding_size)
        self.attention_head = SelfAttentionHead(embedding_size, head_size)
        self.output_layer = nn.Linear(head_size, vocab_size)

    def __call__(self, token_ids):
        positions = mx.arange(token_ids.shape[1])
        embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)
        attended = self.attention_head(embeddings)
        return self.output_layer(attended)

    def loss(self, token_ids, targets):
        logits = self(token_ids)
        batch_size, context_length, vocab_size = logits.shape
        logits = logits.reshape(batch_size * context_length, vocab_size)
        targets = targets.reshape(batch_size * context_length)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def generate(self, token_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            recent_token_ids = token_ids[:, -self.context_length :]
            logits = self(recent_token_ids)
            next_token_id = mx.random.categorical(logits[:, -1, :]).reshape(-1, 1)
            token_ids = mx.concatenate([token_ids, next_token_id], axis=1)
        return token_ids