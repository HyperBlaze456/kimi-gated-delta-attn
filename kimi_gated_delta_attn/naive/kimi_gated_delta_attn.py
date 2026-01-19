import jax
import jax.numpy as jnp
from flax import nnx

from .conv import ShortConvolution
from .norm import FusedRMSNormGated
from .gated_deltanet import recurrent_kda, chunk_kda

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
            # add cache later on. Not in a class, should stay as tuples of array or something
            use_cache: bool = False,

    ):
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                pad_mask = attention_mask[:, 0, :, 0]
            elif attention_mask.ndim == 2:
                pad_mask = attention_mask
            else:
                raise ValueError(
                    "attention_mask must have shape (B, N) or (B, 1, N, 1).",
                )
        else:
            pad_mask = None

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, _ = self.q_conv1d(q, mask=pad_mask)
        k, _ = self.k_conv1d(k, mask=pad_mask)
        v, _ = self.v_conv1d(v, mask=pad_mask)

        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = _kda_gate(
            g,
            self.A_log.get_value(),
            self.head_dim,
            g_bias=self.dt_bias.get_value(),
        )
        beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))

        batch_size, seq_len, _ = hidden_states.shape
        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        mode = "fused_recurrent" if seq_len <= 64 else self.mode

        if mode == "chunk":
            o = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                pad_mask=pad_mask,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            o = recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                pad_mask=pad_mask,
                use_qk_l2norm_in_kernel=True,
            )

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = g.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        o = self.o_norm(o, g)

        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        o = self.o_proj(o)

        if pad_mask is not None:
            o = o * pad_mask[..., None]

        _ = use_cache  # cache plumbing comes later
        return o


def _get_init_key(rngs):
    if rngs is None:
        return jax.random.PRNGKey(0)
    if hasattr(rngs, "params"):
        return rngs.params()
    if callable(rngs):
        return rngs()
    return rngs


def _kda_gate(
    g: jax.Array,
    A_log: jax.Array,
    head_dim: int,
    *,
    g_bias: jax.Array | None = None,
) -> jax.Array:
    """
    Approximate fused_kda_gate from FLA.

    Computes per-head, per-dimension decay factors:
        g_out = exp(-softplus(g + g_bias) * exp(A_log))
    """
    orig_dtype = g.dtype

    if g.ndim == 2:
        g = g[None, :, :]
    if g.ndim == 3:
        num_heads = g.shape[-1] // head_dim
        if g.shape[-1] % head_dim != 0:
            raise ValueError("g last dimension must be divisible by head_dim.")
        g = g.reshape(g.shape[0], g.shape[1], num_heads, head_dim)
    elif g.ndim == 4:
        if g.shape[-1] != head_dim:
            raise ValueError("g last dimension must match head_dim.")
        # Ensure layout is (B, T, H, D)
        if g.shape[2] != A_log.shape[-2] and g.shape[1] == A_log.shape[-2]:
            g = jnp.transpose(g, (0, 2, 1, 3))
    else:
        raise ValueError("g must have shape (B, T, P) or (B, T, H, D).")

    g = g.astype(jnp.float32)
    if g_bias is not None:
        g_bias = g_bias.astype(jnp.float32)
        if g_bias.ndim == 1:
            num_heads = g.shape[2]
            g_bias = g_bias.reshape(1, 1, num_heads, head_dim)
        elif g_bias.ndim == 2:
            g_bias = g_bias.reshape(1, 1, g_bias.shape[0], g_bias.shape[1])
        g = g + g_bias

    A = jnp.exp(A_log.astype(jnp.float32))
    g = jnp.exp(-jax.nn.softplus(g) * A)
    return g.astype(orig_dtype)
