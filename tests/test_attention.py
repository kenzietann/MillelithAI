import mlx.core as mx

from aether.attention import SelfAttentionHead

head = SelfAttentionHead(embedding_size=16, head_size=8)
embeddings = mx.random.normal((2, 5, 16))
output = head(embeddings)
assert output.shape == (2, 5, 8)

changed = mx.array(embeddings)
changed[:, 4, :] = mx.random.normal((2, 16))
changed_output = head(changed)
assert mx.allclose(output[:, :4, :], changed_output[:, :4, :])
print("attention works")