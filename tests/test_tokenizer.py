from aether.tokenizer import CharTokenizer

training_text = "hello world"
tokenizer = CharTokenizer(training_text)

# 8 unique characters: ' ', d , e , h, l, o, r, w
assert tokenizer.vocab_size == 8

# encode then decode should give back exactly we put in
token_ids = tokenizer.encode("hello")
assert len(token_ids) == 5
assert tokenizer.decode(token_ids) == "hello"
assert tokenizer.decode(tokenizer.encode(training_text)) == training_text

# IDs should be stable: sorted characters, so ' ' comes first
assert tokenizer.encode(" ") == [0]

# unknown characters should fail loudly, not silently
try:
    tokenizer.encode("xyz")
    raise AssertionError("expected ValueError for unknown characters")
except ValueError:
    pass

print("tokenizer works")