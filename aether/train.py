import argparse
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

# defaults, every one of them can be overridden from the command line (see parse_args)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--min-learning-rate", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--embedding-size", type=int, default=EMBEDDING_SIZE)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="where to save the model (default: a folder named after the model's settings)",
    )
    args = parser.parse_args()
    if args.checkpoint_dir is None:
        # the name carries the settings, so different experiments never overwrite each other
        args.checkpoint_dir = (
            f"checkpoints/bpe{args.vocab_size}-d{args.embedding_size}"
            f"-l{args.num_layers}-c{args.context_length}"
        )
    return args


def load_or_train_tokenizer(train_text, data_path, vocab_size):
    """Training BPE takes about half a minute, so we keep the result on disk."""
    tokenizer_path = Path("data/processed") / f"{data_path.stem}_bpe_{vocab_size}.json"
    if tokenizer_path.exists():
        return BPETokenizer.from_dict(json.loads(tokenizer_path.read_text()))

    print(f"training a {vocab_size}-token BPE tokenizer (one time only)...")
    tokenizer = BPETokenizer.train(train_text, vocab_size)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.write_text(json.dumps(tokenizer.to_dict()))
    return tokenizer


def main():
    args = parse_args()
    print(f"settings: {vars(args)}")

    text = args.data_path.read_text(encoding="utf-8")
    # keep plain ASCII only for now, so results stay comparable with the character-level runs
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # split the text first and learn the tokenizer from the training part only,
    # otherwise the validation text would leak into the tokenizer
    split_index = int(0.9 * len(text))
    train_text = text[:split_index]
    validation_text = text[split_index:]

    tokenizer = load_or_train_tokenizer(train_text, args.data_path, args.vocab_size)
    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text)
    chars_per_token = len(validation_text) / len(validation_ids)
    print(f"{len(train_ids) + len(validation_ids):,} tokens, vocab size {tokenizer.vocab_size}, "
          f"{chars_per_token:.2f} characters per token")

    # one dict drives both building the model and the checkpoint, so they can't drift apart
    model_config = dict(
        context_length=args.context_length,
        embedding_size=args.embedding_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    model = TransformerModel(tokenizer.vocab_size, **model_config)
    num_parameters = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"{num_parameters / 1e6:.2f}M parameters")

    learning_rate_schedule = make_learning_rate_schedule(
        args.learning_rate, args.min_learning_rate, args.warmup_steps, args.num_steps
    )
    optimizer = optim.AdamW(learning_rate=learning_rate_schedule)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    start_time = time.time()
    for step in range(args.num_steps):
        inputs, targets = get_batch(train_ids, args.batch_size, args.context_length)
        loss, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % args.eval_every == 0:
            train_loss = estimate_loss(
                model, train_ids, args.batch_size, args.context_length, num_batches=20
            )
            validation_loss = estimate_loss(
                model, validation_ids, args.batch_size, args.context_length, num_batches=20
            )
            # loss is per token, but tokens differ between tokenizers. Bits per character
            # is comparable across all of them (lower is better).
            bits_per_character = validation_loss / chars_per_token / math.log(2)
            tokens_seen = (step + 1) * args.batch_size * args.context_length
            tokens_per_second = tokens_seen / (time.time() - start_time)
            print(
                f"step {step}: train {train_loss:.3f}, val {validation_loss:.3f} "
                f"({bits_per_character:.3f} bits/char), "
                f"lr {optimizer.learning_rate.item():.1e}, {tokens_per_second:,.0f} tokens/s"
            )

    # the loop only logs every eval_every steps, so measure the finished model once more,
    # with more batches than usual to keep the number steady
    final_train_loss = estimate_loss(
        model, train_ids, args.batch_size, args.context_length, num_batches=100
    )
    final_validation_loss = estimate_loss(
        model, validation_ids, args.batch_size, args.context_length, num_batches=100
    )
    final_bits_per_character = final_validation_loss / chars_per_token / math.log(2)
    print(
        f"final: train {final_train_loss:.3f}, val {final_validation_loss:.3f} "
        f"({final_bits_per_character:.3f} bits/char)"
    )

    save_checkpoint(model, tokenizer, args.checkpoint_dir, "transformer", model_config)
    print(f"saved to {args.checkpoint_dir}")

    start = mx.array([tokenizer.encode("The ")])
    generated = model.generate(start, max_new_tokens=100)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
