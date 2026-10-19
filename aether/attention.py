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