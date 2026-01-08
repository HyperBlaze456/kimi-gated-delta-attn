import jax
import jax.numpy as jnp

from einops import rearrange # later on convert transpose to rearrange

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

    scores = jnp.einsum("BHND,BHMD->BHNM", q, k) # 이게 N^2로 병목이 됨
    W = scores * L

    if pad_mask is not None:
        W = W * pad_mask[:, None, None, :]

    number = jnp.einsum("BHNM,BHMV->BHNV", W, v)
    denom = jnp.sum(W, axis=-1, keepdims=True)
    out = number / (denom + eps)

    return out

def ssd_masked_linear_attention_scan(q, k, v, a, pad_mask=None, eps=1e-9):
    B, H, N, D = q.shape

    if a.ndim == 1:
        a = a[None, None, :]
    elif a.ndim == 2:
        # Share head
        if a.shape[0] == B and a.shape[1] == N:
             a =a[:, None, :]
        else:
            a = a[None, :, :] # H, N
    a = jnp.broadcast_to(a, (B, H, N))

    if pad_mask is None:
        pad_mask = jnp.ones((B, N), dtype=q.dtype)
    else:
        if pad_mask.ndim == 4:
            pad_mask = pad_mask[:, 0, :, 0] # Only B and N
        pad_mask = pad_mask.astype(q.dtype)

    qT = jnp.transpose(q, (2, 0, 1, 3))
    kT = jnp.transpose(k, (2, 0, 1, 3))
    vT = jnp.transpose(v, (2, 0, 1, 3))
    aT = jnp.transpose(a, (2, 0, 1))      # (N,B,H)
    mT = jnp.transpose(pad_mask, (1, 0))  # (N,B)

    S0 = jnp.zeros((B, H, D, D), dtype=q.dtype) # (B, H, D, Dv) is a robust form, but we expect MHA, non-GQA.
    Z0 = jnp.zeros((B, H, D), dtype=q.dtype) # idk what this is, later update

    def step(carry, inp):
        S, Z = carry

        qt, kt, vt, at, mt = inp

        mt_bh = mt[:, None]                    # (B,1)
        mt_bh = jnp.broadcast_to(mt_bh, at.shape)  # (B,H)

        kt = kt * mt_bh[..., None]
        vt = vt * mt_bh[..., None]

        a_eff = jnp.where(mt_bh > 0, at, 1.0)

        S = a_eff[..., None, None] * S + jnp.einsum("BHD,BHV->BHDV", kt, vt) # recursive outer carry

        Z = a_eff[..., None] * Z + kt

        numerator = jnp.einsum("BHD,BHDV->BHV", qt, S)
        denominator = jnp.einsum("BHD, BHD -> BH", qt, Z) + eps

        out = numerator / denominator[..., None]
        return (S, Z), out

    (Sf, Zf), outT = jax.lax.scan(step, (S0, Z0), (qT, kT, vT, aT, mT))

    out = jnp.transpose(outT, (1, 2, 0, 3))

    return out


def elu_plus_one(x: jax.Array) -> jax.Array:
    return jax.nn.elu(x) + 1.0


def check_equivalence_small():
    key = jax.random.PRNGKey(0)
    B,H,N,D = 2, 3, 8, 4

    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (B,H,N,D))
    k = jax.random.normal(k2, (B,H,N,D))
    v = jax.random.normal(k3, (B,H,N,D))

    # feature map
    q = elu_plus_one(q)
    k = elu_plus_one(k)

    # "실사용" decay: 0~1
    a_logits = jax.random.normal(k4, (B,H,N))
    a = jax.nn.sigmoid(a_logits + 2.0)

    pad_mask = jnp.ones((B,N), dtype=jnp.float32)

    out_exp = ssd_masked_linear_attention_explicit(q, k, v, a, pad_mask=pad_mask, eps=1e-6)
    out_scan = ssd_masked_linear_attention_scan(q, k, v, a, pad_mask=pad_mask, eps=1e-6)

    print("max diff:", jnp.max(jnp.abs(out_exp - out_scan)))

if __name__ == '__main__':
    check_equivalence_small()
