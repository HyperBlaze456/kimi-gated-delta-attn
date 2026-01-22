The architecture solely follows the definition present in [this file](references/modeling.py)

However, core Gated DeltaNet based linear attention functions are inside FLA project, which is super hard to decypher.
They are written in triton, hard to read with all those fused calculations and such.

Moreover, the Kimi Delta Attention uses specialized modules such as ShortConvolution and FusedRMSNormGated.

This architecture explanation would guide through what each module actually does, removing the ambiguity.

# Kimi Delta Attention explained
Kimi Delta Attention is based from the paper [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/pdf/2412.06464)
and also implemented as-is.

The key idea is this: Mamba2 proved how SSMs are actually linear attention, and by adding a gating term linear attention became quite strong.

However, it is still almost impossible to select certain information to keep, thus still not fine enough to keep its performance on extremely long context tasks.

So the paper added Delta rule on top of Mamba2, handling context in a more finely grained manner while keeping the long context capabilities which uses Mamba2's decay term.

## Modules

Even though the core pass is gated delta attention, there are some other modules that hurts the understanding.

There are two modules used, `ShortConvolution` and `FusedRMSNormGated`

### ShortConvolution

ShortConvolution is used to capture short ranged patterns. It was present in Mamba2, and it is used for all Q, K, and V.

It is identical to typical depthwise 1D convolution with causal padding.

But, there are some differences to standard convolution(`nnx.Conv`).

1. There are two activation functions(although they point the same silu())
2. Masks exist, they are applied before anything.
3. Caches are supported
4. For decoding, function step() exists. It naturally pairs when the input x has only 1 value in time dimension with caches.

**Cache and streaming (`step`)**

- Cache stores the most recent `K = kernel_size` inputs, shape `(B, D, K)` where the last axis is time.
- Cache layout is per batch and per channel: `cache[b, d, :]` is the rolling window for channel `d` in batch `b`.
- Cache stores **inputs only** (no outputs). Each `step()` output is computed fresh from the updated cache via a depthwise dot over the time axis (`sum(cache * weight)`), then bias/activation.
- When `__call__` returns a cache, it takes the last `K` timesteps of `x` (or left-pads with zeros if `T < K`) and transposes from `(B, T, D)` to `(B, D, K)`. This matches the `step()` layout.
- In `step()`, the cache is rolled left along the time axis (`shift = -1`), so older values move toward index 0 and index `K-1` becomes the newest slot.
- The new input `x_t` is then written into `cache[:, :, -1]`, overwriting the oldest slot and keeping the window ordered (oldest → newest).
- The output for timestep `t` uses the cache **after** this update, so it always reflects the latest `K` inputs including `x_t`.

**Mini example (K = 3)**

. Start with zeros: `cache[b, d, :] = [0, 0, 0]`
. Receive `x_1`: roll → `[0, 0, 0]`, write → `[0, 0, x_1]`
. Receive `x_2`: roll → `[0, x_1, 0]`, write → `[0, x_1, x_2]`
. Receive `x_3`: roll → `[x_1, x_2, 0]`, write → `[x_1, x_2, x_3]`
. Receive `x_4`: roll → `[x_2, x_3, 0]`, write → `[x_2, x_3, x_4]` (oldest `x_1` dropped)

### FusedRMSNormGated
In the original `triton` implementation, the gating logic is 'fused'.

The JAX implementation does not have all that. It just gates it sequentially. JAX has a strong compiler, so it should be tested whether if the calculations are fused or not after compile.

Fortunately, the logic itself is very straightforward.

We do standard root-mean-square normalization. The gate can be applied before or after.

Then the typical weight and optional bias addition is done.

Overall very simple and easy to understand module, except that further on kernel might be needed.

## Functions


### Main KDA pass

Based on the definition of [gated_deltanet.py](naive/gated_deltanet.py)
 
Per-token recurrence looks like the following: 

```python
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
```

It is identical to DeltaNet's equation except the third line's gating.

Each line underneath the gating term is just Delta attention:

$pred =  k_t^T S_{t-1}$

$delta = v_t - pred$

$S_t = S_{t-1} + beta_t * k_t * delta^T$

Which, when expanded, becomes the same as the original equation in the paper:

$$ S_t = S_{t-1} + \beta_t k_t v_t^T - \beta_t k_t (k_t^T S_{t-1}),                                                                                                                               
      = (I - \beta_t k_t k_t^T) S_{t-1} + \beta_t k_t v_t^T $$

Compared to Gated Delta Attention's equation...

$$ S_{t-1}(\alpha_t (I - \beta_t k_t k_t^T))+ \beta_t k_t v_t^T $$

It is just S_{t-1} gated with $\alpha$ before any processing.

The third line exactly does that.

### Chunk KDA

In the pytorch/triton implementation, it says 'only chunk supports training'.

So I think it is worth noting how exactly chunked KDA function works in detail, and analyze how does the gradient of the function looks like utilizing jax's .grad() transformation.

The first bunch of lines are just shape transposing, padding for chunking, and normalization.

Let's look at the first scanned function, `summary_step`. For starters, we visit how inputs are reshaped and rearranged.

```python
    Np = q.shape[2]
    G = Np // chunk_size
    C = chunk_size
    Dv = v.shape[-1]
    
    q = q.reshape(batch_size, num_heads, G, C, head_dim)
    k = k.reshape(batch_size, num_heads, G, C, head_dim)
    v = v.reshape(batch_size, num_heads, G, C, Dv)
    g = g.reshape(batch_size, num_heads, G, C, head_dim)
    beta = beta.reshape(batch_size, num_heads, G, C)

    kT = jnp.transpose(k, (3, 0, 1, 2, 4))
    vT = jnp.transpose(v, (3, 0, 1, 2, 4))
    gT = jnp.transpose(g, (3, 0, 1, 2, 4))
    betaT = jnp.transpose(beta, (3, 0, 1, 2))

    eye = jnp.eye(head_dim, dtype=q.dtype)
    A0 = jnp.broadcast_to(eye, (batch_size, num_heads, G, head_dim, head_dim))
    B0 = jnp.zeros((batch_size, num_heads, G, head_dim, Dv), dtype=q.dtype)
```

We move the chunk size dimension to the very first axis. (B, H, G, C, D) gets transposed to (C, B, H, G, D).

Due to this, the .scan() works over the token index inside each chunk with length C. All batch/head/chunk groups are processed in parallel.

Let's now actually see the `summary_step` function and how does it accumulate each value.
```python
    def summary_step(carry, inputs):
        A_accum, B_accum = carry
        kt, vt, gt, betat = inputs
        A_accum = _apply_A(gt, kt, betat, A_accum)
        B_accum = _apply_A(gt, kt, betat, B_accum)
        B_accum = B_accum + betat[..., None, None] * jnp.einsum("...d,...v->...dv", kt, vt)
        return (A_accum, B_accum), None
```
- Carry: (A_accum, B_accum)
  - A_accum: starts as identity; becomes the product of per‑token A_t within the chunk. A_t is the unified term to decay and delta update(forget) the previous state.
  - B_accum: starts at 0; becomes the affine offset accumulated from the per‑token “b_t” updates. B_t is what gets added by the delta rule.
- Each step does:
  - A_accum ← A_t · A_accum
  - B_accum ← A_t · B_accum + b_t

We also see that the function _apply_A is also used for calculating both A and B. Let's see how exactly what is going on here.

- b_t = beta_t * (k_t v_t^T)
- `_apply_A()` computes A_t · x without explicitly forming A_t
  - A_t = (I − beta_t k_t k_t^T) · diag(g_t); implemented by first multiplying x by g_t, then applying the rank‑1 update. diag(g_t) term is unique to this Gated DeltaNet.
  - As defined up there, B_t is just A_t time previous B_{t-1}, then update. Remember that updates happen per token, requiring them to be also decayed if they are done parallel via chunks.

- Result after the scan:                                                                                                                                                                                                                                       
  - A_chunk = A_C … A_2 A_1                                                                                                                                                                                                                                  
  - B_chunk = b_C + A_C b_{C-1} + A_C A_{C-1} b_{C-2} + …                                                                                                                                                                                                    
  - So the whole chunk is summarized as S_after = A_chunk · S_before + B_chunk.
  - This is done for every chunk. Now each chunk is 'summarized' to a single state. Further operation to propagate those data throughout all the sequences would be done for the axis 'G'.

The final A_chunk and B_chunk looks like the following.
A_chunk → (B, H, G, D, D)
B_chunk → (B, H, G, D, Dv)



Let's now look at the second function, `chunk_step`.
```python
    A_chunkT = jnp.transpose(A_chunk, (2, 0, 1, 3, 4))
    B_chunkT = jnp.transpose(B_chunk, (2, 0, 1, 3, 4))
```
- A_chunkT = transpose(A_chunk, (2,0,1,3,4)) → (G, B, H, D, D)
- B_chunkT = transpose(B_chunk, (2,0,1,3,4)) → (G, B, H, D, Dv)

We see that the operation is now being done for each chunk.



### KDA gate

We are talking about the function `_kda_gate`

