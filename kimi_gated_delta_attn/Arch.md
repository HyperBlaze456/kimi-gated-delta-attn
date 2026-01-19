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

The definition of pass is present at [gated_deltanet.py](naive/gated_deltanet.py)


