import jax
import jax.numpy as jnp
from flax import nnx


_ACTIVATIONS = {
    None: None,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
}


def _get_init_key(rngs):
    if rngs is None:
        return jax.random.PRNGKey(0)
    if hasattr(rngs, "params"):
        return rngs.params()
    if callable(rngs):
        return rngs()
    return rngs


def _causal_depthwise_conv1d(x, weight, bias):
    # x: (B, T, D), weight: (D, K)
    k = weight.shape[-1]
    x = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))
    kernel = jnp.transpose(weight, (1, 0))[:, None, :]  # (K, 1, D)
    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=weight.shape[0],
    )
    if bias is not None:
        y = y + bias[None, None, :]
    return y


class ShortConvolution(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        dtype: jnp.dtype = jnp.float32,
        kernel_init=jax.nn.initializers.lecun_normal(),
    ):
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation

        key = _get_init_key(rngs)
        weight = kernel_init(key, (hidden_size, kernel_size), dtype)
        self.weight = nnx.Param(weight)
        self.bias = nnx.Param(jnp.zeros((hidden_size,), dtype)) if bias else None

    def step(self, x: jax.Array, cache: jax.Array):
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError("ShortConvolution.step expects a single timestep.")
            x_in = x[:, 0, :]
        else:
            x_in = x
        cache = jnp.roll(cache, shift=-1, axis=-1)
        cache = cache.at[:, :, -1].set(x_in)

        weight = self.weight.get_value().astype(x_in.dtype)
        y = jnp.sum(cache * weight[None, :, :], axis=-1)
        if self.bias is not None:
            y = y + self.bias.get_value().astype(x_in.dtype)
        act = _ACTIVATIONS.get(self.activation)
        if act is not None:
            y = act(y)
        if x.ndim == 3:
            y = y[:, None, :]
        return y, cache

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        cache: jax.Array | None = None,
        output_final_state: bool = False,
    ):
        # TODO: support different sized batches using cu_seqlens and such
        # Also profile

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[..., None]
            x = x * mask

        if cache is not None and x.shape[1] == 1:
            return self.step(x, cache)

        weight = self.weight.get_value().astype(x.dtype)
        bias = self.bias.get_value().astype(x.dtype) if self.bias is not None else None
        y = _causal_depthwise_conv1d(x, weight, bias)

        act = _ACTIVATIONS.get(self.activation)
        if act is not None:
            y = act(y)

        cache_out = None
        if cache is not None or output_final_state:
            tail = x
            if tail.shape[1] < self.kernel_size:
                pad = jnp.zeros(
                    (tail.shape[0], self.kernel_size - tail.shape[1], tail.shape[2]),
                    dtype=tail.dtype,
                )
                tail = jnp.concatenate([pad, tail], axis=1)
            else:
                tail = tail[:, -self.kernel_size :, :]
            cache_out = jnp.transpose(tail, (0, 2, 1))

        return y, cache_out
