import mlx.core as mx
import math

from aether.data import get_batch
from aether.model import BigramModel

vocab_size = 10
model = BigramModel(vocab_size)
token_ids = mx.array([[1, 2, 3], [4, 5, 6]])
logits = model(token_ids)
assert logits.shape == (2, 3, vocab_size)
print("bigram model works")

# loss check: an untrained model guesses randomly, so loss is close to ln(vocab_size)
all_token_ids = list(range(100))
inputs, targets = get_batch(all_token_ids, batch_size=4, context_length=8)
model = BigramModel(vocab_size=100)
loss = model.loss(inputs, targets)
print("loss:", loss.item())
assert abs(loss.item() - math.log(100)) < 1.0
print("loss works")