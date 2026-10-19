import mlx.core as mx
import mlx.nn as nn

from aether.attention import SelfAttentionHead
from aether.transformer import TransformerBlock


class LanguageModel(nn.Module):
    """Shared training and generation logic.

    Subclasses implement __call__ (token IDs -> logits) and set self.context_length
    (None means the model has no limit on how much history it can read).
    """

    def loss(self, token_ids, targets):
        logits = self(token_ids)
        batch_size, context_length, vocab_size = logits.shape
        # cross_entropy wants one row of scores per token, so flatten the batch
        logits = logits.reshape(batch_size * context_length, vocab_size)
        targets = targets.reshape(batch_size * context_length)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def generate(self, token_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            recent_token_ids = token_ids
            if self.context_length is not None:
                # position tables only know context_length positions, so crop the history
                recent_token_ids = token_ids[:, -self.context_length :]
            logits = self(recent_token_ids)
            next_token_id = mx.random.categorical(logits[:, -1, :]).reshape(-1, 1)
            token_ids = mx.concatenate([token_ids, next_token_id], axis=1)
        return token_ids


class BigramModel(LanguageModel):
    def __init__(self, vocab_size):
        super().__init__()
        self.context_length = None
        self.token_table = nn.Embedding(vocab_size, vocab_size)

    def __call__(self, token_ids):
        return self.token_table(token_ids)


class AttentionModel(LanguageModel):
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


class TransformerModel(LanguageModel):
    def __init__(self, vocab_size, context_length, embedding_size, num_heads, num_layers):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_length, embedding_size)
        self.blocks = [TransformerBlock(embedding_size, num_heads) for _ in range(num_layers)]
        self.final_norm = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def __call__(self, token_ids):
        positions = mx.arange(token_ids.shape[1])
        embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            embeddings = block(embeddings)
        return self.output_layer(self.final_norm(embeddings))


MODEL_CLASSES = {
    "bigram": BigramModel,
    "attention": AttentionModel,
    "transformer": TransformerModel,
}
