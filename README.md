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
- [x] Byte pair encoding (BPE) tokenizer, 4096 tokens
- [x] Learning rate warmup, cosine decay and a learning rate sweep
- [x] First 4.8M and 6.9M parameter models (about 10 to 25 minutes on one M4 Max)
- [x] More data (315 MB, about 90 million tokens)
- [x] A 13.8M parameter model (doubling the size gave about 0.07 bits/char)
- [ ] A memory-friendly data pipeline for 1 GB and more of text
- [ ] Bigger models and more data
- [ ] Code and cybersecurity data

Aether can't do anything useful yet. It writes text that looks like English
but doesn't make sense, and we're not going to pretend otherwise.

## Numbers so far

All on the same FineWeb-Edu sample and the same validation text. Loss per
token can't be compared between tokenizers, so everything is in bits per
character (lower is better). The character-level rows were converted from
their original loss per character.

| Model | Tokenizer | Params | Validation bits/char |
|---|---|---|---|
| Bigram | characters | ~0.01M | 3.65 |
| One attention head (64 dim, context 32) | characters | ~0.03M | 3.48 |
| Transformer (4 layers, 128 dim) | characters | 0.82M | ~2.3 |
| Transformer (6 layers, 256 dim), constant lr 1e-3 | characters | 4.80M | ~1.99 |
| Same model, lr 3e-4 with warmup and cosine decay | characters | 4.80M | ~1.86 |
| Transformer (6 layers, 256 dim), same schedule, 70 MB of text | BPE 4096 | 6.85M | ~1.66 |
| Same model, 315 MB of text (one pass) | BPE 4096 | 6.85M | ~1.59 |
| Same, context 128 instead of 64 | BPE 4096 | 6.87M | ~1.60 |
| 6 layers, 384 dim, 315 MB (one pass) | BPE 4096 | 13.82M | 1.52 |

The BPE model has more parameters (its embedding tables are bigger), so part of
the gain over the character-level rows is not only the tokenizer. The 70 MB run
saw its data about 2.2 times, the 315 MB run saw it once.

Training speed on one M4 Max with the 4096-token BPE vocabulary, from
`scripts/benchmark.py` (batch 16):

| Config | Params | Tokens/s | Peak memory | Hours per 1B tokens |
|---|---|---|---|---|
| 6 layers, 384 dim, context 64 | 13.8M | ~33,500 | 0.6 GB | 8 |
| 12 layers, 512 dim, context 128 | 42.1M | ~17,600 | 2.3 GB | 16 |
| 12 layers, 768 dim, context 128 | 91.4M | ~9,000 | 3.5 GB | 31 |
| 24 layers, 1024 dim, context 128 | 310.8M | ~2,800 | 9.6 GB | 101 |

A 1B model needs about 17 GB for weights and optimizer alone, so it doesn't fit
on the 32 GB machine. Bigger models use the GPU more efficiently (parameters
times tokens/s goes up about 1.9x from 14M to 100M, then flattens), so the
real limit on this machine is training time and the amount of data, not memory.
At about 20 tokens per parameter, 91M needs roughly 1.8 billion tokens (about
2.4 days here) and 311M roughly 6.2 billion (about 26 days).

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
  bpe.py          byte pair encoding tokenizer
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

Download the training data (about 315 MB, written to `data/raw/`):

```bash
python scripts/download_dataset.py
```

Train a model. The defaults are the constants at the top of `aether/train.py`,
and every one of them can be changed with a flag (`python -m aether.train --help`
lists them all). The checkpoint goes to a folder named after the settings, for
example `checkpoints/bpe4096-d384-l6-c128/`, so experiments don't overwrite
each other:

```bash
python -m aether.train
python -m aether.train --context-length 128 --embedding-size 384 --num-steps 20000
```

Write some text from a saved checkpoint:

```bash
python -m aether.generate --checkpoint checkpoints/bpe4096-d384-l6-c128 --prompt "The " --max-new-tokens 150
```

(`--max-new-tokens` counts tokens, and one BPE token is about 3.5 characters.)

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
