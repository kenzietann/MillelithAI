import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from aether.data import get_batch
from aether.evaluate import estimate_loss
from aether.model import TransformerModel
from aether.tokenizer import CharTokenizer
from aether.train import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    DATA_PATH,
    EMBEDDING_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
)

LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]
STEPS_PER_RUN = 1500
SEED = 0


def run_one(learning_rate, vocab_size, train_ids, validation_ids, num_steps):
    # same seeds every run, so the only difference between runs is the learning rate
    # (mx seeds the model weights, random seeds which text get_batch picks)
    mx.random.seed(SEED)
    random.seed(SEED)
    model = TransformerModel(
        vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_size=EMBEDDING_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    )
    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    for _ in range(num_steps):
        inputs, targets = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)
        _, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    return estimate_loss(model, validation_ids, BATCH_SIZE, CONTEXT_LENGTH, num_batches=20)


def main():
    text = DATA_PATH.read_text(encoding="utf-8")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)
    split_index = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_index]
    validation_ids = token_ids[split_index:]

    print(f"{STEPS_PER_RUN} steps per run, same model as train.py\n")
    for learning_rate in LEARNING_RATES:
        start_time = time.time()
        validation_loss = run_one(
            learning_rate, tokenizer.vocab_size, train_ids, validation_ids, STEPS_PER_RUN
        )
        elapsed = time.time() - start_time
        print(f"lr {learning_rate:<7} val loss {validation_loss:.3f}  ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
