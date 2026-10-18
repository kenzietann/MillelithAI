from aether.data import get_batch


token_ids = list(range(100))
batch_inputs, batch_targets = get_batch(token_ids, batch_size=4, context_length=8)

assert len(batch_inputs) == len(batch_targets) == 4
for inputs, targets in zip(batch_inputs, batch_targets):
    assert len(inputs) == len(targets) == 8
    # targets is inputs shifted by one, so the overlap must match
    assert inputs[1:] == targets[:-1]
print("get_batch works")