import math

import mlx.core as mx
import mlx.nn as nn


class SelfAttentionHead(nn.Module):
    """One causal self-attention head: each token looks at the tokens before it."""

    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

    def __call__(self, embeddings):
        # embeddings: (batch, context, embedding_size)
        context_length = embeddings.shape[1]

        queries = self.query(embeddings)
        keys = self.key(embeddings)
        values = self.value(embeddings)

        # how well does every query match every key: (batch, context, context)
        scores = (queries @ keys.transpose(0, 2, 1)) / math.sqrt(self.head_size)

        # causal mask: a token may only look at itself and earlier tokens
        allowed = mx.tril(mx.ones((context_length, context_length))) > 0
        scores = mx.where(allowed, scores, -1e9)

        attention_weights = mx.softmax(scores, axis=-1)
        return attention_weights @ values


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, each free to look for a different pattern."""

    def __init__(self, embedding_size, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0, "embedding_size must divide evenly by num_heads"
        head_size = embedding_size // num_heads
        self.heads = [SelfAttentionHead(embedding_size, head_size) for _ in range(num_heads)]
        self.output_projection = nn.Linear(embedding_size, embedding_size)

    def __call__(self, embeddings):
        head_outputs = [head(embeddings) for head in self.heads]
        combined = mx.concatenate(head_outputs, axis=-1)
        return self.output_projection(combined)