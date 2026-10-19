import tempfile

import mlx.core as mx

from aether.checkpoint import load_checkpoint, save_checkpoint
from aether.model import BigramModel, TransformerModel
from aether.tokenizer import CharTokenizer


tokenizer = CharTokenizer("hello world")
token_ids = mx.array([tokenizer.encode("hello")])

# each case: (model type, its settings, the model built from them)
transformer_config = dict(context_length=8, embedding_size=16, num_heads=2, num_layers=2)
cases = [
    ("bigram", {}, BigramModel(tokenizer.vocab_size)),
    ("transformer", transformer_config, TransformerModel(tokenizer.vocab_size, **transformer_config)),
]

for model_type, model_config, model in cases:
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        save_checkpoint(model, tokenizer, checkpoint_dir, model_type, model_config)
        loaded_model, loaded_tokenizer = load_checkpoint(checkpoint_dir)

    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.encode("hello") == tokenizer.encode("hello")

    # same weights in, same scores out
    assert mx.array_equal(model(token_ids), loaded_model(token_ids)), model_type

print("checkpoint works")
