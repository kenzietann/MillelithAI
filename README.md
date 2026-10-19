# Millelith AI

We're building our own language model from scratch. No pretrained weights, no
fine-tuning someone else's model. Just a tokenizer, a transformer, and a
training loop that we write and understand ourselves.

The model is called **Aether**. It's the language model behind **Millelith
Intelligence**, and the long-term goal is a cybersecurity-focused AI that can
help with authorized security work.

## Where things are right now

Still early, but it's a real model now. A small transformer trains from
scratch on a ~70 MB sample of web text and writes English-looking text. Here's
the honest status:

- [x] Character-level tokenizer
- [x] Data loader (random batches of input/target pairs)
- [x] Bigram model, loss, training loop
- [x] Text generation
- [x] Self-attention and a small transformer (multi-head, residual, LayerNorm)
- [x] Train/validation split and averaged evaluation loss
- [x] Training on a real dataset (FineWeb-Edu sample)
- [x] Checkpoints (save and reload any model)
- [x] Benchmark script (tokens per second, memory)
- [ ] Real tokenizer (BPE)
- [x] First 4.8M parameter model (about 25 minutes to train on one M4 Max)
- [ ] Bigger models
- [ ] Code and cybersecurity data

Aether can't do anything useful yet. It writes text that looks like English
but doesn't make sense, and we're not going to pretend otherwise.

## Numbers so far

All on the same FineWeb-Edu sample, character-level, measured on validation
data. Lower loss is better.

| Model | Params | Validation loss |
|---|---|---|
| Bigram | ~0.01M | 2.53 |
| One attention head (64 dim, context 32) | ~0.03M | 2.41 |
| Transformer (4 layers, 128 dim) | 0.82M | ~1.6 |
| Transformer (6 layers, 256 dim), constant lr 1e-3 | 4.80M | ~1.37 |
| Same model, lr 3e-4 with warmup and cosine decay | 4.80M | ~1.29 |

Training speed on one M4 Max, from `scripts/benchmark.py`:

| Config | Params | Tokens/s | Peak memory |
|---|---|---|---|
| 4 layers, 128 dim, context 64 | 0.82M | ~154,000 | 0.24 GB |
| 6 layers, 192 dim, context 64 | 2.71M | ~85,000 | 0.43 GB |
| 6 layers, 256 dim, context 64 | 4.80M | ~56,000 | 0.57 GB |
| 6 layers, 256 dim, context 128 | 4.82M | ~56,000 | 1.12 GB |
| 8 layers, 384 dim, context 128 | 14.31M | ~25,000 | 2.01 GB |

Memory isn't the limit on this machine, training time is.

## Why from scratch?

Mostly because we want to actually understand how this stuff works. Anyone can
download a pretrained model. Fewer people can explain what's happening inside
one. We're learning by building, so the code here is meant to be read and
understood, not just run.

## The plan

| Stage | Size | Goal |
|---|---|---|
| Prototype | 1M to 10M params | Prove the tokenizer and training pipeline work |
| Aether Nano | ~30M | A small but real language model |
| Aether 100M | ~100M | Pretraining on language and security text |
| Aether 300M | ~300M | Scaling experiments |
| Aether 1B | ~1B | Long-term target, if our hardware can handle it |

The bigger sizes are goals, not promises. We'll decide what's realistic based
on real throughput numbers from our machines.

## Tech

- Python
- [MLX](https://github.com/ml-explore/mlx) for compute on Apple Silicon

## Project layout

```
aether/
  tokenizer.py    character-level tokenizer
  data.py         turns token IDs into input/target batches
  attention.py    self-attention head and multi-head attention
  transformer.py  feed-forward layer and transformer block
  model.py        bigram, attention and transformer language models
  train.py        training loop
  evaluate.py     averaged loss on held-out data
  schedule.py     learning rate warmup and cosine decay
  checkpoint.py   save and load models
  generate.py     write text from a saved checkpoint
scripts/          download_dataset.py, benchmark.py, lr_sweep.py
tests/            small tests so we know things actually work
data/             datasets, not tracked in git
checkpoints/      saved model weights, not tracked in git
```

## Running it

You'll need a Mac with Apple Silicon.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy datasets
```

Everything below runs from the repo root.

Run a test (there's one per module in `tests/`):

```bash
python -m tests.test_transformer
```

Download the training data (about 70 MB, written to `data/raw/`):

```bash
python scripts/download_dataset.py
```

Train a model. The model size and settings are the constants at the top of
`aether/train.py`. This saves a checkpoint to `checkpoints/transformer/`:

```bash
python -m aether.train
```

Write some text from the saved checkpoint:

```bash
python -m aether.generate --prompt "The " --max-new-tokens 300
```

See how fast different model sizes train on your machine:

```bash
python -m scripts.benchmark
```

## How we work

- **Start small.** A tiny model that trains beats a huge one that never does.
- **Measure everything.** Loss, tokens per second, memory use.
- **Keep the model separate from any agent tooling.** The language model
  shouldn't depend on pentesting tools.
- **Verify before training on it.** Synthetic or generated data gets checked
  before it goes into a dataset.

Any future security testing will happen only in isolated labs, against
targets we're explicitly authorized to test.

## License

All rights reserved. The code is public so you can read it and follow along,
but you can't copy, modify, or reuse it without permission. See
[LICENSE.md](LICENSE.md).
