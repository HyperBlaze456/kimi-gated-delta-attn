import jax
import jax.numpy as jnp
from flax import nnx


_GATE_ACTIVATIONS = {
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "sigmoid": jax.nn.sigmoid,
    None: None,
}


def _apply_gate(g, activation: str | None):
    act = _GATE_ACTIVATIONS.get(activation)
    if act is None:
        return g
    return act(g)


def rms_norm_gated(
    x: jax.Array,
    g: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None,
    activation: str | None,
    *,
    eps: float = 1e-6,
    residual: jax.Array | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    norm_before_gate: bool = True,
):
    dtype = x.dtype
    if residual is not None:
        if residual_in_fp32:
            x = x.astype(jnp.float32) + residual.astype(jnp.float32)
            residual_out = x
            x = x.astype(dtype)
        else:
            x = x + residual
            residual_out = x
    else:
        residual_out = x

    gate = _apply_gate(g, activation)
    if not norm_before_gate:
        x = x * gate

    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    y = x * jax.lax.rsqrt(var + eps)
    y = y * weight
    if bias is not None:
        y = y + bias

    if norm_before_gate:
        y = y * gate

    if prenorm:
        return y, residual_out
    return y


class FusedRMSNormGated(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str | None = "silu",
        use_bias: bool = False,
        norm_before_gate: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        self.norm_before_gate = norm_before_gate

        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((hidden_size,), dtype=dtype)) if use_bias else None

    def __call__(
        self,
        x: jax.Array,
        g: jax.Array,
        *,
        residual: jax.Array | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ):
        weight = self.weight.get_value().astype(x.dtype)
        bias = self.bias.get_value().astype(x.dtype) if self.bias is not None else None
        return rms_norm_gated(
            x,
            g,
            weight,
            bias,
            self.activation,
            eps=self.eps,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            norm_before_gate=self.norm_before_gate,
        )
