import jax
import jax.numpy as jnp

from einops import rearrange # later on convert transpose to rearrange


def _pad_to_multiple(x, multiple: int, axis: int, pad_value: float):
    """x를 axis 방향으로 multiple 배수 길이가 되도록 pad."""
    n = x.shape[axis]
    pad_len = (-n) % multiple
    if pad_len == 0:
        return x, 0

    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    x = jnp.pad(x, pad_width, constant_values=pad_value)
    return x, pad_len


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

def ssd_linear_attention_chunked(
    q: jax.Array,          # (B, H, N, Dk)   ϕ(q)
    k: jax.Array,          # (B, H, N, Dk)   ϕ(k)
    v: jax.Array,          # (B, H, N, Dv)
    a: jax.Array,          # (B, H, N)  in (0,1],  SSD "마스크 하나"
    pad_mask: jax.Array | None,  # (B, N) with {0,1}
    *,
    chunk_size: int = 64,
    eps: float = 1e-6,
    log_eps: float = 1e-20,
):
    """
    청킹 기반 SSD-masked linear attention (정확한 수식, 청킹만 적용)

    # (수식) SSD 마스크:
    # L[t,s] = Π_{i=s+1..t} a_i   (t>=s), else 0
    #
    # (수식) attention-like form (normalized):
    # W[t,s] = <q_t, k_s> * L[t,s]
    # y_t = (Σ_{s<=t} W[t,s] v_s) / (Σ_{s<=t} W[t,s] + eps)
    #
    # (수식) 상태(state) 관점:
    # S_t = a_t S_{t-1} + k_t ⊗ v_t
    # Z_t = a_t Z_{t-1} + k_t
    # y_t = (q_t^T S_t) / (q_t^T Z_t + eps)
    #
    # 청킹 아이디어(4단계):
    # 1) intra-chunk output: 청크 내부 토큰들끼리만 (C×C matmul)
    # 2) intra-chunk state: 청크 끝(end)까지의 요약 state (S_chunk, Z_chunk)
    # 3) inter-chunk scan: 청크 단위로만 state 전달 (길이 G=N/C)
    # 4) state->output: 각 토큰이 청크 시작 state의 기여를 prefix product로 반영

    반환: (B, H, N, Dv)
    """
    B, H, N, Dk = q.shape
    Dv = v.shape[-1]
    assert k.shape == (B, H, N, Dk)
    assert a.shape == (B, H, N)

    # pad_mask 준비 (패딩 없으면 전부 1)
    if pad_mask is None:
        pad_mask = jnp.ones((B, N), dtype=q.dtype)
    else:
        pad_mask = pad_mask.astype(q.dtype)

    # --- (A) N을 chunk_size 배수로 패딩 ---
    # q,k,v 는 pad 토큰은 0으로, a는 1로, mask는 0으로
    q, pad_len = _pad_to_multiple(q, chunk_size, axis=2, pad_value=0.0)
    k, _       = _pad_to_multiple(k, chunk_size, axis=2, pad_value=0.0)
    v, _       = _pad_to_multiple(v, chunk_size, axis=2, pad_value=0.0)
    a, _       = _pad_to_multiple(a, chunk_size, axis=2, pad_value=1.0)
    pad_mask, _= _pad_to_multiple(pad_mask, chunk_size, axis=1, pad_value=0.0)

    Np = q.shape[2]
    G = Np // chunk_size  # num chunks

    # --- (B) (B,H,N,*) -> (B,H,G,C,*)로 reshape ---
    C = chunk_size
    q = q.reshape(B, H, G, C, Dk)
    k = k.reshape(B, H, G, C, Dk)
    v = v.reshape(B, H, G, C, Dv)
    a = a.reshape(B, H, G, C)
    m = pad_mask.reshape(B, G, C)            # (B,G,C)
    m_bhg = m[:, None, :, :]                 # (B,1,G,C) -> head broadcast 용

    # padding 토큰 처리:
    # - key/value 업데이트 막기: k,v에 mask 곱
    # - decay는 padding에서 1로 두기: state 유지
    k_eff = k * m_bhg[..., None]             # (B,H,G,C,Dk)
    v_eff = v * m_bhg[..., None]             # (B,H,G,C,Dv)
    a_eff = jnp.where(m_bhg > 0, a, 1.0)     # (B,H,G,C)

    # --- (1) intra-chunk용 L_local 만들기 ---
    # (수식)
    # logP[i] = Σ_{t<=i} log(a_t)
    # logL[i,j] = logP[i] - logP[j] = Σ_{t=j+1..i} log(a_t)
    loga = jnp.log(jnp.clip(a_eff, log_eps, 1.0))          # (B,H,G,C)
    logP = jnp.cumsum(loga, axis=-1)                       # (B,H,G,C)

    logL = logP[..., :, None] - logP[..., None, :]         # (B,H,G,C,C)
    tril = jnp.tril(jnp.ones((C, C), dtype=q.dtype))        # (C,C)
    L_local = jnp.exp(logL) * tril[None, None, None, :, :]  # (B,H,G,C,C)

    # prefix product (inclusive): prefix[i] = Π_{t=0..i} a_t
    # (수식) prefix[i] = exp(logP[i])
    prefix = jnp.exp(logP)                                  # (B,H,G,C)
    decay_total = prefix[..., -1]                            # (B,H,G)  Π_{t=0..C-1} a_t

    # --- (1) intra-chunk output: W_local = (qk^T) ∘ L_local ---
    # scores[i,j] = <q_i, k_j>
    scores = jnp.einsum("bhgcd,bhgkd->bhgck", q, k_eff)      # (B,H,G,C,C)
    W_local = scores * L_local                               # (B,H,G,C,C)

    numer_intra = jnp.einsum("bhgck,bhgkv->bhgcv", W_local, v_eff)  # (B,H,G,C,Dv)
    denom_intra = jnp.sum(W_local, axis=-1)                          # (B,H,G,C)

    # --- (2) intra-chunk state 요약: 청크 끝(end)까지의 S_chunk, Z_chunk ---
    # end row의 계수: coeff_end[j] = L_local[end, j] = Π_{t=j+1..end} a_t
    coeff_end = L_local[..., -1, :]                             # (B,H,G,C)

    weighted_k = k_eff * coeff_end[..., None]                   # (B,H,G,C,Dk)
    S_chunk = jnp.einsum("bhgcd,bhgcv->bhgdv", weighted_k, v_eff)  # (B,H,G,Dk,Dv)
    Z_chunk = jnp.sum(weighted_k, axis=3)                        # (B,H,G,Dk)

    # --- (3) inter-chunk scan: 청크 단위로만 state 전달 ---
    # carry:
    # S_g = decay_total_g * S_{g-1} + S_chunk_g
    # Z_g = decay_total_g * Z_{g-1} + Z_chunk_g
    decayT = jnp.transpose(decay_total, (2, 0, 1))              # (G,B,H)
    S_chunkT = jnp.transpose(S_chunk, (2, 0, 1, 3, 4))          # (G,B,H,Dk,Dv)
    Z_chunkT = jnp.transpose(Z_chunk, (2, 0, 1, 3))             # (G,B,H,Dk)

    S0 = jnp.zeros((B, H, Dk, Dv), dtype=q.dtype)
    Z0 = jnp.zeros((B, H, Dk), dtype=q.dtype)

    def step(carry, xs):
        S_prev, Z_prev = carry
        decay_g, Sg, Zg = xs

        # "이 청크 시작 시점" state를 저장
        S_before = S_prev
        Z_before = Z_prev

        # update to end-of-chunk
        S_next = decay_g[..., None, None] * S_prev + Sg
        Z_next = decay_g[..., None] * Z_prev + Zg
        return (S_next, Z_next), (S_before, Z_before)

    (_, _), (S_beforeT, Z_beforeT) = jax.lax.scan(step, (S0, Z0), (decayT, S_chunkT, Z_chunkT))
    S_before = jnp.transpose(S_beforeT, (1, 2, 0, 3, 4))        # (B,H,G,Dk,Dv)
    Z_before = jnp.transpose(Z_beforeT, (1, 2, 0, 3))           # (B,H,G,Dk)

    # --- (4) state->output: 청크 시작 state가 각 토큰에 주는 기여를 prefix로 반영 ---
    # (수식)
    # S_{g,i} = prefix[g,i] * S_before[g] + (intra terms)
    # y_{g,i} 분자에: q_{g,i}^T (prefix * S_before)
    qS_before = jnp.einsum("bhgcd,bhgdv->bhgcv", q, S_before)    # (B,H,G,C,Dv)
    numer_inter = prefix[..., None] * qS_before                   # (B,H,G,C,Dv)

    qZ_before = jnp.einsum("bhgcd,bhgd->bhgc", q, Z_before)      # (B,H,G,C)
    denom_inter = prefix * qZ_before                              # (B,H,G,C)

    numer = numer_intra + numer_inter
    denom = denom_intra + denom_inter + eps
    out = numer / denom[..., None]                                # (B,H,G,C,Dv)

    # query가 padding인 위치는 최종 0으로
    out = out * m_bhg[..., None]                                  # (B,H,G,C,Dv)

    # --- (C) (B,H,G,C,Dv) -> (B,H,N,Dv) 복원 & pad 제거 ---
    out = out.reshape(B, H, Np, Dv)[..., :N, :]
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
