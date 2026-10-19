import json
from pathlib import Path

from aether.model import MODEL_CLASSES
from aether.tokenizer import CharTokenizer

WEIGHTS_FILE = "weights.safetensors"
VOCAB_FILE = "vocab.json"
CONFIG_FILE = "config.json"


def save_checkpoint(model, tokenizer, checkpoint_dir, model_type, model_config):
    """Save everything needed to rebuild the model: weights, vocab, and its settings.

    model_config is the keyword arguments the model was built with, except vocab_size
    (that one comes from the tokenizer).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_weights(str(checkpoint_dir / WEIGHTS_FILE))

    # the vocab is just the characters in ID order, that's all we need to rebuild the tokenizer
    vocab = [tokenizer.id_to_char[token_id] for token_id in range(tokenizer.vocab_size)]
    with open(checkpoint_dir / VOCAB_FILE, "w") as vocab_file:
        json.dump(vocab, vocab_file)

    config = {"model_type": model_type, "model_config": model_config}
    with open(checkpoint_dir / CONFIG_FILE, "w") as config_file:
        json.dump(config, config_file, indent=2)


def load_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / VOCAB_FILE) as vocab_file:
        vocab = json.load(vocab_file)
    # the vocab is already sorted, so rebuilding from it gives the exact same IDs
    tokenizer = CharTokenizer("".join(vocab))

    with open(checkpoint_dir / CONFIG_FILE) as config_file:
        config = json.load(config_file)

    model_class = MODEL_CLASSES[config["model_type"]]
    model = model_class(tokenizer.vocab_size, **config["model_config"])
    model.load_weights(str(checkpoint_dir / WEIGHTS_FILE))
    return model, tokenizer
