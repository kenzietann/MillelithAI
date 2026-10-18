# Millelith AI

We're building our own language model from scratch. No pretrained weights, no
fine-tuning someone else's model. Just a tokenizer, a transformer, and a
training loop that we write and understand ourselves.

The model is called **Aether**. It's the language model behind **Millelith
Intelligence**, and the long-term goal is a cybersecurity-focused AI that can
help with authorized security work.

## Where things are right now

Very early. Honestly, we're at the "teach it to guess the next character"
stage. Here's the real status:

- [x] Character-level tokenizer
- [x] Data loader (random batches of input/target pairs)
- [ ] First model (bigram)
- [ ] Training loop with loss logging
- [ ] Text generation
- [ ] Small transformer
- [ ] Real tokenizer (BPE)
- [ ] Training on real data

Aether can't do anything useful yet, and we're not going to pretend it can.

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
aether/       the model code (tokenizer, data loading, and more to come)
tests/        small tests so we know things actually work
configs/      model and training configs (coming soon)
data/         datasets, not tracked in git
checkpoints/  saved model weights, not tracked in git
docs/         notes and write-ups
```

## Running it

You'll need a Mac with Apple Silicon.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy
```

Then run the tests from the repo root:

```bash
python -m tests.test_data
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
