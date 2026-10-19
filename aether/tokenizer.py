class CharTokenizer:
    """Turns text into token IDs, one ID per unique character."""

    def __init__(self, text):
        # sorted so the IDs are the same every run, otherwise set order is random
        unique_characters = sorted(set(text))
        self.char_to_id = {char: token_id for token_id, char in enumerate(unique_characters)}
        self.id_to_char = {token_id: char for char, token_id in self.char_to_id.items()}
        self.vocab_size = len(unique_characters)

    def encode(self, text):
        """Text -> list of token IDs."""
        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as error:
            # better than a bare KeyError, tells you what actually went wrong
            raise ValueError(f"Character {error} is not in the vocabulary") from error

    def decode(self, token_ids):
        """List of token IDs -> text."""                                            
        return "".join(self.id_to_char[token_id] for token_id in token_ids)

    def to_dict(self):
        vocab = [self.id_to_char[token_id] for token_id in range(self.vocab_size)]
        return {"type": "char", "vocab": vocab}

    @classmethod
    def from_dict(cls, data):
        # the vocab is already sorted, so rebuilding from it gives the exact same IDs
        return cls("".join(data["vocab"]))
