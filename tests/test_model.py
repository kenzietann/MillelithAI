import mlx.core as mx

from aether.model import BigramModel

vocab_size = 10
model = BigramModel(vocab_size)

token_ids = mx.array([[1, 2, 3], [4, 5, 6]])
logits = model(token_ids)

print(logits.shape)
assert logits.shape == (2, 3, vocab_size)
print("bigram model works")