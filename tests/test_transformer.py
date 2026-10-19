import mlx.core as mx

from aether.model import TransformerModel
from aether.transformer import TransformerBlock

# a block must keep the shape: (batch, context, embedding) in, same shape out
block = TransformerBlock(embedding_size=32, num_heads=4)
embeddings = mx.random.normal((2, 6, 32))
assert block(embeddings).shape == (2, 6, 32)

model = TransformerModel(vocab_size=50, context_length=16, embedding_size=32, num_heads=4, num_layers=2)
token_ids = mx.random.randint(0, 50, (2, 16))
assert model(token_ids).shape == (2, 16, 50)

# causality for the whole model: changing the last token must not change earlier scores
changed = mx.array(token_ids)
changed[:, -1] = (token_ids[:, -1] + 1) % 50
assert mx.allclose(model(token_ids)[:, :-1, :], model(changed)[:, :-1, :], atol=1e-5)

# generating past the context window must still work
generated = model.generate(mx.array([[0]]), max_new_tokens=40)
assert generated.shape == (1, 41)
print("transformer works")
