import jax
import jax.numpy as jnp


def _pad_to_multiple(x: jax.Array, multiple: int, axis: int, pad_value: float):
    n = x.shape[axis]
    pad_len = (-n) % multiple
    if pad_len == 0:
        return x, 0

    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    x = jnp.pad(x, pad_width, constant_values=pad_value)
    return x, pad_len


def _maybe_transpose_bnhd_to_bhnd(x: jax.Array) -> tuple[jax.Array, bool]:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {x.shape}.")
    if x.shape[1] > x.shape[2]:
        return jnp.transpose(x, (0, 2, 1, 3)), True
    return x, False


def _prepare_pad_mask(
    pad_mask: jax.Array | None,
    batch_size: int,
    seq_len: int,
    dtype: jnp.dtype,
) -> jax.Array:
    if pad_mask is None:
        return jnp.ones((batch_size, seq_len), dtype=dtype)
    if pad_mask.ndim == 4:
        pad_mask = pad_mask[:, 0, :, 0]
    if pad_mask.shape != (batch_size, seq_len):
        raise ValueError(f"pad_mask shape {pad_mask.shape} does not match (B, N)=({batch_size}, {seq_len}).")
    return pad_mask.astype(dtype)


def _broadcast_g(
    g: jax.Array | None,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> jax.Array:
    if g is None:
        return jnp.ones((batch_size, num_heads, seq_len, head_dim), dtype=dtype)
    if g.ndim == 4 and g.shape[-1] == 1:
        g = g[..., 0]
    if g.ndim == 1:
        g = g[None, None, :, None]
    elif g.ndim == 2:
        if g.shape == (batch_size, seq_len):
            g = g[:, None, :, None]
        elif g.shape == (num_heads, seq_len):
            g = g[None, :, :, None]
        else:
            raise ValueError(f"Unsupported g shape {g.shape}.")
    elif g.ndim == 3:
        if g.shape == (batch_size, num_heads, seq_len):
            g = g[..., None]
        elif g.shape == (batch_size, seq_len, num_heads):
            g = jnp.transpose(g, (0, 2, 1))[..., None]
        else:
            raise ValueError(f"Unsupported g shape {g.shape}.")
    elif g.ndim == 4:
        if g.shape == (batch_size, num_heads, seq_len, head_dim):
            pass
        elif g.shape == (batch_size, seq_len, num_heads, head_dim):
            g = jnp.transpose(g, (0, 2, 1, 3))
        else:
            raise ValueError(f"Unsupported g shape {g.shape}.")
    else:
        raise ValueError(f"Unsupported g shape {g.shape}.")
    return jnp.broadcast_to(g, (batch_size, num_heads, seq_len, head_dim)).astype(dtype)


def _broadcast_beta(
    beta: jax.Array | None,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    dtype: jnp.dtype,
) -> jax.Array:
    if beta is None:
        return jnp.ones((batch_size, num_heads, seq_len), dtype=dtype)
    if beta.ndim == 4 and beta.shape[-1] == 1:
        beta = beta[..., 0]
    if beta.ndim == 1:
        beta = beta[None, None, :]
    elif beta.ndim == 2:
        if beta.shape == (batch_size, seq_len):
            beta = beta[:, None, :]
        elif beta.shape == (num_heads, seq_len):
            beta = beta[None, :, :]
        else:
            raise ValueError(f"Unsupported beta shape {beta.shape}.")
    elif beta.ndim == 3:
        if beta.shape == (batch_size, num_heads, seq_len):
            pass
        elif beta.shape == (batch_size, seq_len, num_heads):
            beta = jnp.transpose(beta, (0, 2, 1))
        else:
            raise ValueError(f"Unsupported beta shape {beta.shape}.")
    else:
        raise ValueError(f"Unsupported beta shape {beta.shape}.")
    return jnp.broadcast_to(beta, (batch_size, num_heads, seq_len)).astype(dtype)


def _l2_normalize(x: jax.Array, eps: float) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _apply_A(g_t: jax.Array, k_t: jax.Array, beta_t: jax.Array, x: jax.Array) -> jax.Array:
    # Applies A_t = (I - beta_t * k_t k_t^T) * diag(g_t) without forming A_t.
    x1 = g_t[..., None] * x
    proj = jnp.einsum("...d,...dk->...k", k_t, x1)
    update = jnp.einsum("...d,...k->...dk", k_t, proj)
    return x1 - beta_t[..., None, None] * update


def recurrent_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None,
    beta: jax.Array | None,
    *,
    initial_state: jax.Array | None = None,
    pad_mask: jax.Array | None = None,
    eps: float = 1e-6,
    use_qk_l2norm_in_kernel: bool = False,
    output_final_state: bool = False,
    **_unused,
):
    """
    Per-token recurrent delta network. Ops not fused -> pallas kernel write might be needed.

    Expected shapes (preferred): q,k,v = (B, H, N, D).
    If given (B, N, H, D), tensors are transposed heuristically.
    g: (B,H,N,D) or broadcastable to it. beta: (B,H,N) or broadcastable to it.
    """
    q, q_transposed = _maybe_transpose_bnhd_to_bhnd(q)
    k, k_transposed = _maybe_transpose_bnhd_to_bhnd(k)
    v, v_transposed = _maybe_transpose_bnhd_to_bhnd(v)
    if q_transposed != k_transposed or q_transposed != v_transposed:
        raise ValueError("q, k, v must share the same layout (BHND or BNHD).")

    batch_size, num_heads, seq_len, head_dim = q.shape
    if k.shape != (batch_size, num_heads, seq_len, head_dim):
        raise ValueError(f"k shape {k.shape} does not match q shape {q.shape}.")
    if v.shape[:3] != (batch_size, num_heads, seq_len):
        raise ValueError(f"v shape {v.shape} does not match q shape {q.shape}.")

    g = _broadcast_g(g, batch_size, num_heads, seq_len, head_dim, q.dtype)
    beta = _broadcast_beta(beta, batch_size, num_heads, seq_len, q.dtype)
    mask = _prepare_pad_mask(pad_mask, batch_size, seq_len, q.dtype)

    if use_qk_l2norm_in_kernel:
        q = _l2_normalize(q, eps)
        k = _l2_normalize(k, eps)

    mask_bh = mask[:, None, :, None]
    q = q * mask_bh
    k = k * mask_bh
    v = v * mask_bh
    beta = beta * mask[:, None, :]
    g = jnp.where(mask_bh > 0, g, 1.0)

    qT = jnp.transpose(q, (2, 0, 1, 3))
    kT = jnp.transpose(k, (2, 0, 1, 3))
    vT = jnp.transpose(v, (2, 0, 1, 3))
    gT = jnp.transpose(g, (2, 0, 1, 3))
    betaT = jnp.transpose(beta, (2, 0, 1))

    if initial_state is None:
        S0 = jnp.zeros((batch_size, num_heads, head_dim, v.shape[-1]), dtype=q.dtype)
    else:
        S0 = initial_state.astype(q.dtype)

    def step(S, inputs):
        qt, kt, vt, gt, betat = inputs
        S_base = gt[..., None] * S
        pred = jnp.einsum("bhd,bhdv->bhv", kt, S_base)
        delta = vt - pred
        S = S_base + betat[..., None, None] * jnp.einsum("bhd,bhv->bhdv", kt, delta)
        out = jnp.einsum("bhd,bhdv->bhv", qt, S)
        return S, out

    Sf, outT = jax.lax.scan(step, S0, (qT, kT, vT, gT, betaT))
    out = jnp.transpose(outT, (1, 2, 0, 3))

    if q_transposed:
        out = jnp.transpose(out, (0, 2, 1, 3))

    if output_final_state:
        return out, Sf
    return out


def chunk_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None,
    beta: jax.Array | None,
    *,
    initial_state: jax.Array | None = None,
    pad_mask: jax.Array | None = None,
    chunk_size: int = 64,
    eps: float = 1e-6,
    use_qk_l2norm_in_kernel: bool = False,
    output_final_state: bool = False,
    **_unused,
):
    """
    Chunked delta network recurrence.

    Expected shapes (preferred): q,k,v = (B, H, N, D).
    If given (B, N, H, D), tensors are transposed heuristically.
    """
    q, q_transposed = _maybe_transpose_bnhd_to_bhnd(q)
    k, k_transposed = _maybe_transpose_bnhd_to_bhnd(k)
    v, v_transposed = _maybe_transpose_bnhd_to_bhnd(v)
    if q_transposed != k_transposed or q_transposed != v_transposed:
        raise ValueError("q, k, v must share the same layout (BHND or BNHD).")

    batch_size, num_heads, seq_len, head_dim = q.shape
    if k.shape != (batch_size, num_heads, seq_len, head_dim):
        raise ValueError(f"k shape {k.shape} does not match q shape {q.shape}.")
    if v.shape[:3] != (batch_size, num_heads, seq_len):
        raise ValueError(f"v shape {v.shape} does not match q shape {q.shape}.")

    g = _broadcast_g(g, batch_size, num_heads, seq_len, head_dim, q.dtype)
    beta = _broadcast_beta(beta, batch_size, num_heads, seq_len, q.dtype)
    mask = _prepare_pad_mask(pad_mask, batch_size, seq_len, q.dtype)

    if use_qk_l2norm_in_kernel:
        q = _l2_normalize(q, eps)
        k = _l2_normalize(k, eps)

    mask_bh = mask[:, None, :, None]
    q = q * mask_bh
    k = k * mask_bh
    v = v * mask_bh
    beta = beta * mask[:, None, :]
    g = jnp.where(mask_bh > 0, g, 1.0)

    q, _ = _pad_to_multiple(q, chunk_size, axis=2, pad_value=0.0)
    k, _ = _pad_to_multiple(k, chunk_size, axis=2, pad_value=0.0)
    v, _ = _pad_to_multiple(v, chunk_size, axis=2, pad_value=0.0)
    g, _ = _pad_to_multiple(g, chunk_size, axis=2, pad_value=1.0)
    beta, _ = _pad_to_multiple(beta, chunk_size, axis=2, pad_value=0.0)

    Np = q.shape[2]
    G = Np // chunk_size
    C = chunk_size
    Dv = v.shape[-1]

    q = q.reshape(batch_size, num_heads, G, C, head_dim)
    k = k.reshape(batch_size, num_heads, G, C, head_dim)
    v = v.reshape(batch_size, num_heads, G, C, Dv)
    g = g.reshape(batch_size, num_heads, G, C, head_dim)
    beta = beta.reshape(batch_size, num_heads, G, C)

    kT = jnp.transpose(k, (3, 0, 1, 2, 4)) # C, B, H, G, D
    vT = jnp.transpose(v, (3, 0, 1, 2, 4))
    gT = jnp.transpose(g, (3, 0, 1, 2, 4))
    betaT = jnp.transpose(beta, (3, 0, 1, 2))

    eye = jnp.eye(head_dim, dtype=q.dtype)
    A0 = jnp.broadcast_to(eye, (batch_size, num_heads, G, head_dim, head_dim))
    B0 = jnp.zeros((batch_size, num_heads, G, head_dim, Dv), dtype=q.dtype)

    def summary_step(carry, inputs):
        A_accum, B_accum = carry
        kt, vt, gt, betat = inputs
        A_accum = _apply_A(gt, kt, betat, A_accum)
        B_accum = _apply_A(gt, kt, betat, B_accum)
        B_accum = B_accum + betat[..., None, None] * jnp.einsum("...d,...v->...dv", kt, vt)
        return (A_accum, B_accum), None
    # Scan in chunk dimension?

    (A_chunk, B_chunk), _ = jax.lax.scan(summary_step, (A0, B0), (kT, vT, gT, betaT))

    A_chunkT = jnp.transpose(A_chunk, (2, 0, 1, 3, 4))
    B_chunkT = jnp.transpose(B_chunk, (2, 0, 1, 3, 4))

    if initial_state is None:
        S0 = jnp.zeros((batch_size, num_heads, head_dim, Dv), dtype=q.dtype)
    else:
        S0 = initial_state.astype(q.dtype)

    def chunk_step(S_prev, inputs):
        A_g, B_g = inputs
        S_before = S_prev
        S_next = jnp.einsum("bhij,bhjv->bhiv", A_g, S_prev) + B_g
        return S_next, S_before

    Sf, S_beforeT = jax.lax.scan(chunk_step, S0, (A_chunkT, B_chunkT))
    S_before = jnp.transpose(S_beforeT, (1, 2, 0, 3, 4))

    qT = jnp.transpose(q, (2, 3, 0, 1, 4))
    kT = jnp.transpose(k, (2, 3, 0, 1, 4))
    vT = jnp.transpose(v, (2, 3, 0, 1, 4))
    gT = jnp.transpose(g, (2, 3, 0, 1, 4))
    betaT = jnp.transpose(beta, (2, 3, 0, 1))
    S_beforeT = jnp.transpose(S_before, (2, 0, 1, 3, 4))

    def scan_chunk(S_init, q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk):
        def step(S, inputs):
            qt, kt, vt, gt, betat = inputs
            S_base = gt[..., None] * S
            pred = jnp.einsum("bhd,bhdv->bhv", kt, S_base)
            delta = vt - pred
            S = S_base + betat[..., None, None] * jnp.einsum("bhd,bhv->bhdv", kt, delta)
            out = jnp.einsum("bhd,bhdv->bhv", qt, S)
            return S, out

        _, out = jax.lax.scan(step, S_init, (q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk))
        return out

    outT = jax.vmap(scan_chunk, in_axes=(0, 0, 0, 0, 0, 0))(S_beforeT, qT, kT, vT, gT, betaT)
    out = jnp.transpose(outT, (2, 3, 0, 1, 4))
    out = out.reshape(batch_size, num_heads, Np, Dv)[..., :seq_len, :]

    if q_transposed:
        out = jnp.transpose(out, (0, 2, 1, 3))

    if output_final_state:
        return out, Sf
    return out

