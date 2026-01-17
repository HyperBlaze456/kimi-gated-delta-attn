import jax
import jax.numpy as jnp
from flax import nnx

from .conv import ShortConvolution
from .norm import FusedRMSNormGated
from .gated_deltanet import fused_recurrent_kda, chunk_kda

class KimiDeltaAttention(nnx.Module):
    def __init__(
            self,
            hidden_size: int,
            conv_size: int,
            head_dim: int,
            num_heads: int,
            head_k_dim: int,
            num_k_heads: int,
            layer_idx: int,
            *,
            rms_norm_eps: float = 1e-6,
            rngs: nnx.Rngs | None = None,
    ):

        self.mode = "fused_recurrent"
        self.hidden_size = hidden_size
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.num_k_heads = num_k_heads

        self.layer_idx = layer_idx

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        linear_kwargs = {"use_bias": False}
        if rngs is not None:
            linear_kwargs["rngs"] = rngs

        self.q_proj = nnx.Linear(self.hidden_size, projection_k_size, **linear_kwargs)
        self.k_proj = nnx.Linear(self.hidden_size, projection_k_size, **linear_kwargs)
        self.v_proj = nnx.Linear(self.hidden_size, projection_size, **linear_kwargs)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
            rngs=rngs,
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
            rngs=rngs,
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
            rngs=rngs,
        )

        key = _get_init_key(rngs)
        a_key = key
        a_init = jax.random.uniform(
            a_key,
            (self.num_heads,),
            minval=1.0,
            maxval=16.0,
            dtype=jnp.float32,
        )
        self.A_log = nnx.Param(jnp.log(a_init).reshape(1, 1, self.num_heads, 1))

        self.f_a_proj = nnx.Linear(self.hidden_size, self.head_dim, **linear_kwargs)
        self.f_b_proj = nnx.Linear(self.head_dim, projection_size, **linear_kwargs)

        self.dt_bias = nnx.Param(jnp.zeros((projection_size,), dtype=jnp.float32))

        self.b_proj = nnx.Linear(self.hidden_size, self.num_heads, **linear_kwargs)

        self.g_a_proj = nnx.Linear(self.hidden_size, self.head_dim, **linear_kwargs)
        self.g_b_proj = nnx.Linear(self.head_dim, projection_size, **linear_kwargs)

        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=rms_norm_eps,
            activation="sigmoid",
        )

        self.o_proj = nnx.Linear(projection_size, self.hidden_size, **linear_kwargs)

    def __call__(
            self,
            hidden_states: jax.Array,
            attention_mask: jax.Array | None = None,
            # add cache

    ):

def _get_init_key(rngs):
    if rngs is None:
        return jax.random.PRNGKey(0)
    if hasattr(rngs, "params"):
        return rngs.params()
    if callable(rngs):
        return rngs()
    return rngs
