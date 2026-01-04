import jax
import jax.numpy as jnp
from flax import nnx

class KimiDeltaAttention(nnx.Module):
    def __init__(
            self,
            hidden_size: int,
            conv_kernel_size: int,
            head_dim: int,
            num_heads: int,
    ):
        # no mode config
        self.hidden_size = hidden_size
        self.conv_size = conv_kernel_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

