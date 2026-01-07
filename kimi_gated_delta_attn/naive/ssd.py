import jax
import jax.numpy as jnp

from einops import rearrange

def make_ssd_L(a: jax.Array, eps: float = 1e-9) -> jax.Array:
    if a.ndim == 1:
        a = a[None, None, :]
    elif a.ndim == 2:
        a = a[None, :, :]

    B, H, N = a.shape
    loga = jnp.log(jnp.clip(a, eps, 1.0))
    logP = jnp.cumsum(loga, axis=-1)

    logL = logP[..., :, None] - logP[..., None, :] # B, H, N, N

    causal = jnp.tril(jnp.ones((N,N), dtype=a.dtype))
    L = jnp.exp(logL) * causal[None, None, :, :]

    return L

def ssd_masked_linear_attention_explicit(q, k, v, a, pad_mask=None, eps=1e-9):
    B, H, N, D = q.shape
    L = make_ssd_L(a)

    scores = jnp.einsum("BHND,BHMD->BHNM", q, k)
    W = scores * L

    if pad_mask is not None:
        W = W * pad_mask[:, None, None, :]

    number = jnp.einsum("BHNM,BHMV->BHNV", q, v)
    denom = jnp.sum(W, axis=-1, keepdims=True)
    out = number / (denom + eps)

    return out

