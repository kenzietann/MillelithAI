import random
import mlx.core as mx


def get_pair(token_ids, start, context_length):
    inputs = token_ids[start : start + context_length]
    targets = token_ids[start + 1 : start + 1 + context_length]
    return inputs, targets

def get_batch(token_ids, batch_size, context_length):
    max_start = len(token_ids) - context_length - 1
    batch_inputs = []
    batch_targets = []
    for _ in range(batch_size):
        start = random.randint(0, max_start)
        inputs, targets = get_pair(token_ids, start, context_length)
        batch_inputs.append(inputs)
        batch_targets.append(targets)
    return mx.array(batch_inputs), mx.array(batch_targets)