import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from aether.model import TransformerModel

VOCAB_SIZE = 4096  # matches the BPE tokenizer in train.py
BATCH_SIZE = 32
WARMUP_STEPS = 5
MEASURED_STEPS = 30

# (name, embedding_size, num_heads, num_layers, context_length)
CONFIGS = [
    ("current", 256, 8, 6, 64),
    ("longer context", 256, 8, 6, 128),
    ("wider", 384, 8, 6, 128),
    ("wider and deeper", 384, 8, 8, 128),
    ("big", 512, 8, 8, 128),
]


def benchmark(name, embedding_size, num_heads, num_layers, context_length):
    model = TransformerModel(
        VOCAB_SIZE,
        context_length=context_length,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    num_parameters = sum(p.size for _, p in tree_flatten(model.parameters()))
    optimizer = optim.AdamW(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    # random tokens are fine here, speed doesn't depend on what the tokens are
    inputs = mx.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length))
    targets = mx.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length))

    def train_step():
        _, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    # the first steps are slow because MLX is still compiling, don't count them
    for _ in range(WARMUP_STEPS):
        train_step()

    mx.reset_peak_memory()
    start_time = time.time()
    for _ in range(MEASURED_STEPS):
        train_step()
    elapsed = time.time() - start_time

    tokens_per_second = MEASURED_STEPS * BATCH_SIZE * context_length / elapsed
    peak_memory_gb = mx.get_peak_memory() / 1e9
    print(
        f"{name:<16} {num_parameters / 1e6:>6.2f}M params   "
        f"{tokens_per_second:>9,.0f} tokens/s   {peak_memory_gb:>5.2f} GB peak"
    )


def main():
    print(f"batch size {BATCH_SIZE}, {MEASURED_STEPS} measured steps per config\n")
    for config in CONFIGS:
        benchmark(*config)


if __name__ == "__main__":
    main()
