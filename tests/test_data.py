import mlx.core as mx
from aether.data import get_batch


token_ids = list(range(100))
batch_inputs, batch_targets = get_batch(token_ids, batch_size=4, context_length=8)

assert batch_inputs.shape == (4, 8)
assert batch_targets.shape == (4, 8)

# inputs without its first column should match targets without its last column
assert mx.array_equal(batch_inputs[:, 1:], batch_targets[:, :-1])

# stress test the max_start boundary: a bigger batch makes an off-by-one show up
big_inputs, big_targets = get_batch(token_ids, batch_size=10000, context_length=8)
assert big_inputs.shape == (10000, 8)
assert big_targets.shape == (10000, 8)
assert mx.array_equal(big_inputs[:, 1:], big_targets[:, :-1])

print("get_batch works")