from aether.data import get_batch


def estimate_loss(model, token_ids, batch_size, context_length, num_batches):
    """Average the loss over many random batches, so one lucky batch doesnt fool us"""
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch(token_ids, batch_size, context_length)
        losses.append(model.loss(inputs, targets).item())
    return sum(losses) / len(losses)