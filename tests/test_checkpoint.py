import tempfile
import mlx.core as mx

from aether.checkpoint import load_checkpoint, save_checkpoint
from aether.model import BigramModel
from aether.tokenizer import CharTokenizer


tokenizer = CharTokenizer("hello world")
model = BigramModel(tokenizer.vocab_size)
token_ids = mx.array([tokenizer.encode("hello")])

with tempfile.TemporaryDirectory() as checkpoint_dir:
    save_checkpoint(model, tokenizer, checkpoint_dir)
    loaded_model, loaded_tokenizer = load_checkpoint(checkpoint_dir)

assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
assert loaded_tokenizer.encode("hello") == tokenizer.encode("hello")

# same weights in, some scores out
assert mx.array_equal(model(token_ids), loaded_model(token_ids))
print("checkpoint works")