import json
import tempfile
from pathlib import Path

import mlx.core as mx

from aether.bpe import BPETokenizer
from aether.checkpoint import load_checkpoint, save_checkpoint
from aether.model import BigramModel, TransformerModel
from aether.tokenizer import CharTokenizer


corpus = "hello world, hello there. the world says hello to the world."
char_tokenizer = CharTokenizer(corpus)
bpe_tokenizer = BPETokenizer.train(corpus * 4, vocab_size=280)

transformer_config = dict(context_length=8, embedding_size=16, num_heads=2, num_layers=2)

# each case: (model type, its settings, tokenizer, the model built from them)
cases = [
    ("bigram", {}, char_tokenizer, BigramModel(char_tokenizer.vocab_size)),
    ("transformer", transformer_config, char_tokenizer, TransformerModel(char_tokenizer.vocab_size, **transformer_config)),
    ("transformer", transformer_config, bpe_tokenizer, TransformerModel(bpe_tokenizer.vocab_size, **transformer_config)),
]

for model_type, model_config, tokenizer, model in cases:
    token_ids = mx.array([tokenizer.encode("hello")])  # shorter than context_length=8
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        save_checkpoint(model, tokenizer, checkpoint_dir, model_type, model_config)
        loaded_model, loaded_tokenizer = load_checkpoint(checkpoint_dir)

    assert type(loaded_tokenizer) is type(tokenizer)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.encode("hello world") == tokenizer.encode("hello world")

    # same weights in, same scores out
    assert mx.array_equal(model(token_ids), loaded_model(token_ids)), model_type

# older checkpoints (before tokenizer.json existed) stored only vocab.json and must still load
with tempfile.TemporaryDirectory() as checkpoint_dir:
    model = BigramModel(char_tokenizer.vocab_size)
    save_checkpoint(model, char_tokenizer, checkpoint_dir, "bigram", {})
    Path(checkpoint_dir, "tokenizer.json").unlink()
    with open(Path(checkpoint_dir, "vocab.json"), "w") as vocab_file:
        json.dump(char_tokenizer.to_dict()["vocab"], vocab_file)
    _, legacy_tokenizer = load_checkpoint(checkpoint_dir)
    assert legacy_tokenizer.encode("hello") == char_tokenizer.encode("hello")

print("checkpoint works")
