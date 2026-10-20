import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from aether.model import TransformerModel

VOCAB_SIZE = 4096  # matches the BPE tokenizer in train.py
BATCH_SIZE = 16  # smaller than in training so the big models fit, tokens/s hardly depends on it
WARMUP_STEPS = 5
MEASURED_STEPS = 20

# Training keeps weights, gradients and the two AdamW states in fp32: 16 bytes per parameter.
# Configs above this estimate are skipped, so the benchmark can't push the Mac into swapping.
BYTES_PER_PARAMETER = 16
MAX_ESTIMATED_GB = 16

# (name, embedding_size, num_heads, num_layers, context_length)
CONFIGS = [
    ("14M (best so far)", 384, 8, 6, 64),
    ("40M", 512, 8, 12, 128),
    ("100M", 768, 12, 12, 128),
    ("300M", 1024, 16, 24, 128),
    ("1B", 1536, 24, 36, 128),
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
    estimated_gb = num_parameters * BYTES_PER_PARAMETER / 1e9
    if estimated_gb > MAX_ESTIMATED_GB:
        print(
            f"{name:<18} {num_parameters / 1e6:>7.1f}M params   skipped: needs about "
            f"{estimated_gb:.0f} GB for weights and optimizer alone (limit {MAX_ESTIMATED_GB} GB)"
        )
        return
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
    # time to read one pass over the data in this model, handy for planning
    hours_for_one_billion_tokens = 1e9 / tokens_per_second / 3600
    print(
        f"{name:<18} {num_parameters / 1e6:>7.1f}M params   "
        f"{tokens_per_second:>8,.0f} tokens/s   {peak_memory_gb:>5.1f} GB peak   "
        f"{hours_for_one_billion_tokens:>6.0f} h per 1B tokens"
    )


def main():
    print(f"batch size {BATCH_SIZE}, {MEASURED_STEPS} measured steps per config\n")
    for config in CONFIGS:
        benchmark(*config)


if __name__ == "__main__":
    main()
