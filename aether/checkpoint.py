import json
from pathlib import Path

from aether.model import BigramModel
from aether.tokenizer import CharTokenizer

WEIGHTS_FILE = "weights.safetensors"
VOCAB_FILE = "vocab.json"


def save_checkpoint(model, tokenizer, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_weights(str(checkpoint_dir / WEIGHTS_FILE))

    # the vocab is just the characters in ID order, thats all we need to rebuild the tokenizer
    vocab = [tokenizer.id_to_char[token_id] for token_id in range(tokenizer.vocab_size)]
    with open(checkpoint_dir / VOCAB_FILE, "w") as vocab_file:
        json.dump(vocab, vocab_file)


def load_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / VOCAB_FILE) as vocab_file:
        vocab = json.load(vocab_file)

    # the vocab is already sorted, so rebuilding from it gives the exact same IDs
    tokenizer = CharTokenizer("".join(vocab))

    model = BigramModel(tokenizer.vocab_size)
    model.load_weights(str(checkpoint_dir / WEIGHTS_FILE))
    return model, tokenizer
