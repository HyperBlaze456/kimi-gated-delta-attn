import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

from .ssd import ssd_masked_linear_attention_scan

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

class SSDLinearAttention(nnx.Module):
    """
    SSD-masked Linear Attention (no chunking).

    - 기존 linear attention에 SSD mask L을 "딸-깍" 적용한 것과 동치
    - 단, L을 직접 만들지 않고 scan으로 계산 (O(N))

    Args:
        hidden_size, num_heads, head_dim, eps: 동일
        decay_bias_shift: a_t를 sigmoid로 만들 때 초기 기억 길이 조절 (큰 값 => a가 1에 가까움)
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        decay_bias_shift: float = 2.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.decay_bias_shift = decay_bias_shift

        projection_size = num_heads * head_dim

        self.q_proj = nnx.Linear(hidden_size, projection_size, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, projection_size, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, projection_size, use_bias=False, rngs=rngs)

        self.o_proj = nnx.Linear(projection_size, hidden_size, use_bias=False, rngs=rngs)

        # "실사용 가능한" SSD mask 하나(a_t)를 만들기 위한 projection
        # a_logits: (B,N,H)
        self.a_proj = nnx.Linear(hidden_size, num_heads, use_bias=True, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        *,
        ssd_a: jax.Array | None = None,
    ) -> jax.Array:
        """
        x: (B,N,D)
        mask: padding mask (B,N) or (B,1,N,1) with {0,1}
        ssd_a: SSD "마스크 하나" (옵션)
               - (N,) or (B,N) or (B,H,N) or (B,N,H)
               - 값 범위는 (0,1] 권장
        """
        B, N, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # feature map φ
        q = elu_plus_one(q)
        k = elu_plus_one(k)

        # padding mask 정리 -> (B,N)
        if mask is not None:
            if mask.ndim == 4:
                pad_mask = mask[:, 0, :, 0]
            else:
                pad_mask = mask
            pad_mask = pad_mask.astype(x.dtype)
        else:
            pad_mask = None

        # SSD mask 하나(a_t) 준비
        if ssd_a is None:
            # (수식) a_t = sigmoid(W_a x_t + b + shift)  in (0,1)
            a_logits = self.a_proj(x)  # (B,N,H)
            a = jax.nn.sigmoid(a_logits + self.decay_bias_shift)  # (B,N,H)
            a = rearrange(a, "b n h -> b h n")  # (B,H,N)
        else:
            a = ssd_a
            if a.ndim == 1:
                a = a[None, None, :]            # (1,1,N)
            elif a.ndim == 2:
                a = a[:, None, :]               # (B,1,N)  (head 공유)
            elif a.ndim == 3:
                # (B,N,H) 들어오면 transpose
                if a.shape[1] == N and a.shape[2] == self.num_heads:
                    a = rearrange(a, "b n h -> b h n")
            a = jnp.broadcast_to(a, (B, self.num_heads, N))

        out = ssd_masked_linear_attention_scan(q, k, v, a, pad_mask=pad_mask, eps=self.eps)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.o_proj(out)

        # padding 위치 출력까지 깔끔히 0으로 만들고 싶으면:
        if pad_mask is not None:
            out = out * pad_mask[..., None]
        return out
