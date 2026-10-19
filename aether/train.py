import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from aether.bpe import BPETokenizer
from aether.checkpoint import save_checkpoint
from aether.data import get_batch
from aether.evaluate import estimate_loss
from aether.model import TransformerModel
from aether.schedule import make_learning_rate_schedule

DATA_PATH = Path("data/raw/fineweb_edu_300mb.txt")
VOCAB_SIZE = 4096
BATCH_SIZE = 32
CONTEXT_LENGTH = 64
NUM_STEPS = 40000  # about one pass over 300 MB of text (roughly 80 million tokens)
EVAL_EVERY = 5000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_STEPS = 1000
EMBEDDING_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 6
CHECKPOINT_DIR = "checkpoints/transformer-bpe-300mb"


def load_or_train_tokenizer(train_text):
    """Training BPE takes about half a minute, so we keep the result on disk."""
    tokenizer_path = Path("data/processed") / f"{DATA_PATH.stem}_bpe_{VOCAB_SIZE}.json"
    if tokenizer_path.exists():
        return BPETokenizer.from_dict(json.loads(tokenizer_path.read_text()))

    print(f"training a {VOCAB_SIZE}-token BPE tokenizer (one time only)...")
    tokenizer = BPETokenizer.train(train_text, VOCAB_SIZE)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.write_text(json.dumps(tokenizer.to_dict()))
    return tokenizer


def main():
    text = DATA_PATH.read_text(encoding="utf-8")
    # keep plain ASCII only for now, so results stay comparable with the character-level runs
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # split the text first and learn the tokenizer from the training part only,
    # otherwise the validation text would leak into the tokenizer
    split_index = int(0.9 * len(text))
    train_text = text[:split_index]
    validation_text = text[split_index:]

    tokenizer = load_or_train_tokenizer(train_text)
    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text)
    chars_per_token = len(validation_text) / len(validation_ids)
    print(f"{len(train_ids) + len(validation_ids):,} tokens, vocab size {tokenizer.vocab_size}, "
          f"{chars_per_token:.2f} characters per token")

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
            # loss is per token, but tokens differ between tokenizers. Bits per character
            # is comparable across all of them (lower is better).
            bits_per_character = validation_loss / chars_per_token / math.log(2)
            tokens_seen = (step + 1) * BATCH_SIZE * CONTEXT_LENGTH
            tokens_per_second = tokens_seen / (time.time() - start_time)
            print(
                f"step {step}: train {train_loss:.3f}, val {validation_loss:.3f} "
                f"({bits_per_character:.3f} bits/char), "
                f"lr {optimizer.learning_rate.item():.1e}, {tokens_per_second:,.0f} tokens/s"
            )

    save_checkpoint(model, tokenizer, CHECKPOINT_DIR, "transformer", model_config)

    start = mx.array([tokenizer.encode("The ")])
    generated = model.generate(start, max_new_tokens=100)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
