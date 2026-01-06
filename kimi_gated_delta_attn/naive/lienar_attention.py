import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

def elu_plus_one(x: jax.Array) -> jax.Array:
    # uses this positive lock funciton instead of softmax
    return jax.nn.elu(x) + 1.0


class LinearAttention(nnx.Module):
    """
    Linear Attention with ELU+1 feature map.

    Uses the kernel trick to achieve O(N) complexity instead of O(N^2).

    Args:
        hidden_size: Input hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        eps: Small constant for numerical stability
        rngs: Random number generators for initialization
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps

        projection_size = num_heads * head_dim

        self.q_proj = nnx.Linear(hidden_size, projection_size, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, projection_size, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, projection_size, use_bias=False, rngs=rngs)

        self.o_proj = nnx.Linear(projection_size, hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """
        Forward pass for linear attention.

        Args:
            x: Input tensor of shape (B, N, D) where
               B = batch size, N = sequence length, D = hidden dimension
            mask: Optional multiplicative mask of shape (B, N) or (B, 1, N, 1)
                  Values should be 0 for masked positions and 1 for valid positions

        Returns:
            Output tensor of shape (B, N, D)
        """
        bs, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)


        q = elu_plus_one(q)
        k = elu_plus_one(k)

        if mask is not None:
            # Reshape mask to (B, 1, N, 1) for broadcasting
            if mask.ndim == 2:
                mask = mask[:, None, :, None]
            k = k * mask
            v = v * mask

        # Linear attention: O(N) complexity using the kernel trick
        # Instead of computing attention weights explicitly, we compute:
        # output = (Q @ (K^T @ V)) / (Q @ sum(K^T))

        # KV matrix: (B, H, D, D) = (B, H, D, N) @ (B, H, N, D)
        kv_mat = jnp.einsum("bhnd,bhnv->bhdv", k, v)

        # Numerator: Q @ KV -> (B, H, N, D)
        numerator = jnp.einsum("bhnd,bhdv->bhnv", q, kv_mat)

        # Denominator: Q @ sum(K) for normalization -> (B, H, N, 1)
        k_sum = k.sum(axis=2, keepdims=True)  # (B, H, 1, D)
        denominator = jnp.einsum("bhnd,bhkd->bhnk", q, k_sum)  # (B, H, N, 1)

        # Normalize with epsilon for numerical stability
        output = numerator / (denominator + self.eps)

        output = rearrange(output, "b h n d -> b n (h d)")

        output = self.o_proj(output)

        return output



