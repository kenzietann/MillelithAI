import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from aether.checkpoint import save_checkpoint
from aether.data import get_batch
from aether.evaluate import estimate_loss
from aether.model import TransformerModel
from aether.schedule import make_learning_rate_schedule
from aether.tokenizer import CharTokenizer

DATA_PATH = Path("data/raw/fineweb_edu_sample.txt")
BATCH_SIZE = 32
CONTEXT_LENGTH = 64
NUM_STEPS = 30000
EVAL_EVERY = 2500
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_STEPS = 1000
EMBEDDING_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 6


def main():
    text = DATA_PATH.read_text(encoding="utf-8")
    # keep plain ASCII only for now, so the vocab stays small
    text = text.encode("ascii", errors="ignore").decode("ascii")

    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    split_index = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_index]
    validation_ids = token_ids[split_index:]
    print(f"{len(token_ids):,} tokens, vocab size {tokenizer.vocab_size}")

    # one dict drives both building the model and the checkpoint, so they can't drift apart
    model_config = dict(
        context_length=CONTEXT_LENGTH,
        embedding_size=EMBEDDING_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    )
    model = TransformerModel(tokenizer.vocab_size, **model_config)
    num_parameters = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"{num_parameters / 1e6:.2f}M parameters")

    learning_rate_schedule = make_learning_rate_schedule(
        LEARNING_RATE, MIN_LEARNING_RATE, WARMUP_STEPS, NUM_STEPS
    )
    optimizer = optim.AdamW(learning_rate=learning_rate_schedule)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    start_time = time.time()
    for step in range(NUM_STEPS):
        inputs, targets = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)
        loss, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % EVAL_EVERY == 0:
            train_loss = estimate_loss(model, train_ids, BATCH_SIZE, CONTEXT_LENGTH, num_batches=20)
            validation_loss = estimate_loss(model, validation_ids, BATCH_SIZE, CONTEXT_LENGTH, num_batches=20)
            tokens_seen = (step + 1) * BATCH_SIZE * CONTEXT_LENGTH
            tokens_per_second = tokens_seen / (time.time() - start_time)
            print(
                f"step {step}: train {train_loss:.3f}, val {validation_loss:.3f}, "
                f"lr {optimizer.learning_rate.item():.1e}, {tokens_per_second:,.0f} tokens/s"
            )

    save_checkpoint(model, tokenizer, "checkpoints/transformer", "transformer", model_config)

    start = mx.array([[0]])
    generated = model.generate(start, max_new_tokens=300)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
