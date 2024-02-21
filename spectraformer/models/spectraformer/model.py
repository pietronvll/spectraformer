import chex
import flax.linen as nn
import jax
import numpy as np


class LinearProjection(nn.Module):
    """Linear embedding projection"""

    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embedding_dim, use_bias=True)(x)
        return x


def test_LinearProjection():
    """Test LinearProjection"""
    x = np.random.randn(1, 2, 3, 100)
    model = LinearProjection(128)
    variables = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(variables, x)
    chex.assert_shape(y, (1, 2, 3, 128))
    chex.assert_shape(variables["params"]["Dense_0"]["kernel"], (100, 128))


test_LinearProjection()


class FFBlock(nn.Module):
    """Feed-forward block for transformer model"""

    embedding_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(4 * self.embedding_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.embedding_dim)(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""

    embedding_dim: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, attn_mask, training: bool):
        # Multi-head attention
        x_norm = nn.LayerNorm()(x)
        x_norm = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.embedding_dim
        )(x_norm, deterministic=not training)
        x = x_norm
        # x = x + nn.Dense(self.embedding_dim)(x_norm) # Residual connection
        x_norm = nn.LayerNorm()(x)
        x_norm = FFBlock(self.embedding_dim, self.dropout_rate)(
            x_norm, training=training
        )
        x = x + nn.Dense(self.embedding_dim)(x_norm)
        return x


class SpectraFormer(nn.Module):
    embedding_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, counts, wave_number, wave_number_mask, training: bool = False):
        # wave_number_mask is a 1D boolean array of shape num_wave_numbers
        emb_counts = LinearProjection(self.embedding_dim)(
            counts
        )  # [batch_size, num_wave_numbers, 1]
        emb_wave_number = LinearProjection(self.embedding_dim)(
            wave_number
        )  # [num_wave_numbers, 1]
        x = (
            emb_counts + emb_wave_number
        )  # [batch_size, num_wave_numbers, embedding_dim]
        attn_mask = nn.make_attention_mask(
            wave_number_mask, wave_number_mask, extra_batch_dims=x.ndim - 2
        )
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                self.embedding_dim, self.num_heads, self.dropout_rate
            )(x, attn_mask, training=training)
        x = nn.LayerNorm()(x)
        x = nn.Dense(1)(x)
        return x
