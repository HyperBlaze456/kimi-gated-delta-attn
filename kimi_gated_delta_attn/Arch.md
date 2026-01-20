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

So I think it is worth noting how exactly chunked KDA function works in detail, and analyze how does the gradient of the function looks like.

  - 입력 레이아웃 정렬/검증 후 g, beta, pad_mask를 브로드캐스트/정규화합니다. q,k,v는 BHND 형식이 기본이고 BNHD면 전치합니다. pad_mask는 (B,N)로 맞춥니다. (kimi_gated_delta_attn/naive/gated_deltanet.py:224-249)                                               
  - 옵션으로 q/k L2 정규화 후, 마스크 적용: q,k,v는 0으로, beta는 0으로, g는 1로 만들어 패딩 토큰이 상태에 영향 못 주게 합니다. (240-249)                                                                                                                        
  - 시퀀스를 chunk_size 배수로 패딩합니다. q,k,v는 0, g는 1, beta는 0으로 패딩됩니다. (251-255)                                                                                                                                                                  
  - (B,H,N,D) → (B,H,G,C,…)로 리쉐이프(G=chunk 수)하고, time-major로 transpose합니다. (262-271)                                                                                                                                                                  
  - summary_step: 각 chunk 내부 토큰을 스캔해서 A_chunk, B_chunk를 만듭니다. 여기서 _apply_A는                                                                                                                                                                   
    A_t = (I - beta_t * k_t k_t^T) * diag(g_t) 를 명시적으로 만들지 않고 누적에 적용합니다.                                                                                                                                                                      
    B는 beta * k * v^T 항을 누적합니다. (summary_step:277-285, _apply_A:116-121)                                                                                                                                                                                 
  - chunk_step: chunk 단위로 S_next = A_chunk * S_prev + B_chunk를 스캔해 각 chunk 시작 상태 S_before를 계산합니다. (295-302)                                                                                                                                    
  - 출력은 다시 원래 shape로 합치고 패딩된 길이를 잘라냅니다. 입력이 BNHD였으면 복원 전치합니다. (325-329)
  - output_final_state=True면 chunk-level scan의 최종 상태 Sf를 함께 반환합니다. (331-332)

  배경: 원래 per-token recurrence (recurrent_kda)                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                 
  - 상태 S_t는 (B,H,D,Dv) 형태이고, 토큰마다 업데이트됩니다.                                                                                                                                                                                                     
  - 업데이트식은 아래와 같습니다. ( kimi_gated_delta_attn/naive/gated_deltanet.py:183-189 )                                                                                                                                                                      
      - S_base = diag(g_t) * S_{t-1}                                                                                                                                                                                                                             
      - pred = k_t^T * S_base  (코드: einsum("bhd,bhdv->bhv", kt, S_base))                                                                                                                                                                                       
      - delta = v_t - pred                                                                                                                                                                                                                                       
      - S_t = S_base + beta_t * (k_t * delta^T)                                                                                                                                                                                                                  
        (코드: einsum("bhd,bhv->bhdv", kt, delta))                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                 
  이걸 토큰마다 직접 스캔하면 O(N) 순차인데, chunk_kda는 chunk 단위로 병렬화하려고 summary_step와 chunk_step을 둡니다.                                                                                                                                           
                                                                                                                                                                                                                                                                 
  ———                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                 
  1) summary_step — chunk 내부를 요약해 A_chunk, B_chunk 생성                                                                                                                                                                                                    
  코드 위치: summary_step (277-285), _apply_A (116-121)                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                 
  핵심 아이디어:                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                 
  - per-token update는 “선형 변환 + 저차 업데이트” 구조라서, chunk 안의 여러 토큰을 합치면                                                                                                                                                                       
                                                                                                                                                                                                                                                                 
    S_after_chunk = A_chunk * S_before_chunk + B_chunk                                                                                                                                                                                                           
    형태로 요약됩니다.                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                 
  summary_step는 이 A_chunk, B_chunk를 토큰을 따라 누적해서 만들어요.                                                                                                                                                                                            
                                                                                                                                                                                                                                                                 
  정의:                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                 
  - _apply_A(gt, kt, betat, X)는 다음 선형변환을 X에 적용:                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                 
    A_t = (I - beta_t * k_t k_t^T) * diag(g_t)
    (코드에서는 행렬을 만들지 않고, x1 = g_t * X, proj = k_t^T x1, x1 - beta*k_t*proj로 처리)                                                                                                                                                                    
                                                                                                                                                                                                                                                                 
  summary_step 안의 누적:                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                 
  - A_accum는 A_t들을 곱한 결과:
                                                                                                                                                                                                                                                                 
    A_accum ← A_t * A_accum                                                                                                                                                                                                                                      
  - B_accum는 현재까지 누적된 bias:                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                 
    B_accum ← A_t * B_accum + beta_t * k_t * v_t^T                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                 
  그래서 chunk 내부 토큰들을 scan하면,                                                                                                                                                                                                                           
  “이 chunk를 통째로 통과했을 때 상태가 어떻게 변하는지”가 A_chunk, B_chunk로 요약됩니다.                                                                                                                                                                        
                                                                                                                                                                                                                                                                 
  ———                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                 
  2) chunk_step — chunk 경계 상태를 빠르게 이동                                                                                                                                                                                                                  
  코드 위치: chunk_step (295-299), jax.lax.scan (301)                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                 
  이제 각 chunk는 하나의 선형 변환처럼 취급 가능:                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                 
  S_next = A_chunk * S_prev + B_chunk                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                 
  chunk_step는 이걸 chunk 순서대로 스캔해서:                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                 
  - S_before = 각 chunk의 시작 상태                                                                                                                                                                                                                              
  - Sf = 마지막 chunk 이후 최종 상태                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                 
  을 계산합니다.                                                                                                                                                                                                                                                 
  즉, per-token이 아니라 chunk 단위로 상태를 전개하는 단계입니다.                                                                                                                                                                                                
                                                                                                                                                                                                                                                                 
  ———                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                 
  왜 두 단계가 필요하나                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                 
  - summary_step는 chunk 내부 요약 (A/B 생성)                                                                                                                                                                                                                    
  - chunk_step는 chunk 경계 상태 전개                                                                                                                                                                                                                            
  - 이후에야 S_before를 초기값으로 per-token recurrence를 chunk마다 병렬 처리할 수 있음 (vmap)                                                                                                                                                                   
                                                                                                                                                                                                                                                                 
  즉, summary_step → chunk_step 덕분에 전체 시퀀스를 순차 처리하지 않고,                                                                                                                                                                                         
  chunk 단위 병렬성을 확보합니다.

### KDA gate

We are talking about the function `_kda_gate`

