import mlx.nn as nn

from aether.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """A small two-layer network applied to every token on its own."""

    def __init__(self, embedding_size):
        super().__init__()
        self.expand = nn.Linear(embedding_size, 4 * embedding_size)
        self.project = nn.Linear(4 * embedding_size, embedding_size)

    def __call__(self, embeddings):
        return self.project(nn.gelu(self.expand(embeddings)))

class TransformerBlock(nn.Module):
    """Attention (tokens talk to each other), then feed-forward (each token thinks alone)."""

    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.attention = MultiHeadAttention(embedding_size, num_heads)
        self.feed_forward_norm = nn.LayerNorm(embedding_size)
        self.feed_forward = FeedForward(embedding_size)

    def __call__(self, embeddings):
        embeddings = embeddings + self.attention(self.attention_norm(embeddings))
        embeddings = embeddings + self.feed_forward(self.feed_forward_norm(embeddings))
        return embeddings