from collections import Counter

from aether.bpe import BPETokenizer, count_pairs, merge_pair, split_into_words, train_merges

# --- the two building blocks ---
assert count_pairs([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
assert merge_pair([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
# overlapping case: scanning left to right, a token can't be used twice
assert merge_pair([1, 1, 1], (1, 1), 99) == [99, 1]

# --- splitting text into words must never lose or change any text ---
messy = "Hello,  world!\n\tsnake_case 123 café ☃ end "
assert "".join(split_into_words(messy)) == messy


# --- the fast trainer must learn exactly what a simple, obviously correct one learns ---
def train_merges_slowly(word_counts, num_merges):
    words = {word: list(word.encode("utf-8")) for word in word_counts}
    merges = []
    for merge_number in range(num_merges):
        pair_counts = Counter()
        for word, token_ids in words.items():
            for pair, count in count_pairs(token_ids).items():
                pair_counts[pair] += count * word_counts[word]
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        for word in words:
            words[word] = merge_pair(words[word], best_pair, 256 + merge_number)
    return merges


text = (
    "the quick brown fox jumps over the lazy dog. the lower the lowest low-level lows. "
    "security is not an afterthought; the secure thing is the simple thing. "
) * 5 + "tokens, tokenizers and tokenization tokenize text. " * 3
word_counts = Counter(split_into_words(text))
assert train_merges(word_counts, 60) == train_merges_slowly(word_counts, 60)

# --- round trip: decode(encode(x)) == x for anything, including text it never saw ---
tokenizer = BPETokenizer.train(text, vocab_size=300)
assert tokenizer.vocab_size == 300
for sample in [text, messy, "", " ", "café ☃ \U0001f600", "snake_case__ and\r\nwindows"]:
    assert tokenizer.decode(tokenizer.encode(sample)) == sample, repr(sample)

# --- merging actually compresses text it was trained on ---
assert len(tokenizer.encode(text)) < len(text.encode("utf-8")) / 2

# --- saving and loading gives the same tokenizer ---
loaded = BPETokenizer.from_dict(tokenizer.to_dict())
assert loaded.encode(text) == tokenizer.encode(text)
assert loaded.vocab_size == tokenizer.vocab_size

print("bpe works")
