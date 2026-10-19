import json
import re
from collections import Counter, defaultdict

BASE_VOCAB_SIZE = 256  # every possible byte is a token, so any text can be encoded

# Split text into word-like chunks (a word keeps its leading space, e.g. " the").
# Merges never cross chunk boundaries, which keeps tokens sensible and training fast.
WORD_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\w\s]|_)+|\s+")


def split_into_words(text):
    return WORD_PATTERN.findall(text)


def count_pairs(token_ids):
    """Count how often each pair of neighbouring tokens appears."""
    pair_counts = {}
    for pair in zip(token_ids, token_ids[1:]):
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def merge_pair(token_ids, pair, new_token_id):
    """Replace every occurrence of `pair` with `new_token_id`, scanning left to right."""
    merged = []
    position = 0
    while position < len(token_ids):
        is_match = (
            position < len(token_ids) - 1
            and token_ids[position] == pair[0]
            and token_ids[position + 1] == pair[1]
        )
        if is_match:
            merged.append(new_token_id)
            position += 2
        else:
            merged.append(token_ids[position])
            position += 1
    return merged


def train_merges(word_counts, num_merges):
    """Learn which token pairs to merge, most frequent first.

    word_counts maps each distinct word to how often it appears in the text. Working on
    distinct words (instead of the whole text) and only updating the words a merge touches
    is what makes this fast enough for tens of megabytes.
    """
    words = [list(word.encode("utf-8")) for word in word_counts]
    frequencies = list(word_counts.values())

    pair_counts = Counter()
    pair_to_word_indices = defaultdict(set)
    for index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequencies[index]
            pair_to_word_indices[pair].add(index)

    merges = []
    for merge_number in range(num_merges):
        if not pair_counts:
            break
        # highest count wins, ties are broken by the pair itself so results are repeatable
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_token_id = BASE_VOCAB_SIZE + merge_number
        merges.append(best_pair)

        for index in list(pair_to_word_indices[best_pair]):
            word = words[index]
            new_word = merge_pair(word, best_pair, new_token_id)
            if new_word == word:
                continue  # stale index: this word no longer contains the pair
            frequency = frequencies[index]
            for pair in zip(word, word[1:]):
                pair_counts[pair] -= frequency
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += frequency
                pair_to_word_indices[pair].add(index)
            words[index] = new_word
        del pair_to_word_indices[best_pair]
    return merges


class BPETokenizer:
    """Byte pair encoding: frequent neighbouring tokens get fused into one bigger token."""

    def __init__(self, merges):
        self.merges = [tuple(pair) for pair in merges]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        # the bytes each token stands for, needed to turn IDs back into text
        self.token_bytes = [bytes([byte]) for byte in range(BASE_VOCAB_SIZE)]
        for first, second in self.merges:
            self.token_bytes.append(self.token_bytes[first] + self.token_bytes[second])
        self.vocab_size = len(self.token_bytes)
        self._word_cache = {}

    @classmethod
    def train(cls, text, vocab_size):
        word_counts = Counter(split_into_words(text))
        return cls(train_merges(word_counts, vocab_size - BASE_VOCAB_SIZE))

    def _encode_word(self, word):
        if word in self._word_cache:
            return self._word_cache[word]
        token_ids = list(word.encode("utf-8"))
        while len(token_ids) >= 2:
            # apply the merge that was learned first, because earlier merges are more basic
            best_pair = min(
                zip(token_ids, token_ids[1:]),
                key=lambda pair: self.merge_ranks.get(pair, float("inf")),
            )
            if best_pair not in self.merge_ranks:
                break
            new_token_id = BASE_VOCAB_SIZE + self.merge_ranks[best_pair]
            token_ids = merge_pair(token_ids, best_pair, new_token_id)
        self._word_cache[word] = token_ids
        return token_ids

    def encode(self, text):
        token_ids = []
        for word in split_into_words(text):
            token_ids.extend(self._encode_word(word))
        return token_ids

    def decode(self, token_ids):
        raw_bytes = b"".join(self.token_bytes[token_id] for token_id in token_ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def to_dict(self):
        return {"type": "bpe", "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, data):
        return cls(data["merges"])
