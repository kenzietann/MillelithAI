import json
from pathlib import Path

from aether.bpe import BPETokenizer
from aether.model import MODEL_CLASSES
from aether.tokenizer import CharTokenizer

WEIGHTS_FILE = "weights.safetensors"
TOKENIZER_FILE = "tokenizer.json"
LEGACY_VOCAB_FILE = "vocab.json"  # older checkpoints only stored the characters
CONFIG_FILE = "config.json"


def tokenizer_from_dict(data):
    if data["type"] == "char":
        return CharTokenizer.from_dict(data)
    if data["type"] == "bpe":
        return BPETokenizer.from_dict(data)
    raise ValueError(f"Unknown tokenizer type: {data['type']}")


def save_checkpoint(model, tokenizer, checkpoint_dir, model_type, model_config):
    """Save everything needed to rebuild the model: weights, tokenizer, and its settings.

    model_config is the keyword arguments the model was built with, except vocab_size
    (that one comes from the tokenizer).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_weights(str(checkpoint_dir / WEIGHTS_FILE))

    with open(checkpoint_dir / TOKENIZER_FILE, "w") as tokenizer_file:
        json.dump(tokenizer.to_dict(), tokenizer_file)

    config = {"model_type": model_type, "model_config": model_config}
    with open(checkpoint_dir / CONFIG_FILE, "w") as config_file:
        json.dump(config, config_file, indent=2)


def load_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    if (checkpoint_dir / TOKENIZER_FILE).exists():
        with open(checkpoint_dir / TOKENIZER_FILE) as tokenizer_file:
            tokenizer = tokenizer_from_dict(json.load(tokenizer_file))
    else:
        with open(checkpoint_dir / LEGACY_VOCAB_FILE) as vocab_file:
            tokenizer = CharTokenizer.from_dict({"vocab": json.load(vocab_file)})

    with open(checkpoint_dir / CONFIG_FILE) as config_file:
        config = json.load(config_file)

    model_class = MODEL_CLASSES[config["model_type"]]
    model = model_class(tokenizer.vocab_size, **config["model_config"])
    model.load_weights(str(checkpoint_dir / WEIGHTS_FILE))
    return model, tokenizer
