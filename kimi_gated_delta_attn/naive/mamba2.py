"""
Mamba2 Implementation using SSD (State Space Duality)

Mamba2 is an efficient sequence model that unifies SSMs and Attention through
the State Space Duality framework. Key differences from Mamba1:
  - A matrix: Scalar × Identity (vs Diagonal in Mamba1)
  - Larger state dimension: 64-128 (vs 16 in Mamba1)
  - Multi-head structure similar to attention
  - 2-8x faster training via tensor cores

Architecture:
  Input (B, L, D)
    ↓
  in_proj → [z, x, B, C, dt]
    ↓
  conv1d(x) → x'  (short convolution for local context)
    ↓
  SSD core: y = SSD(q=C, k=B, v=x', a=exp(-dt*A))
    ↓
  RMSNorm(y) * SiLU(z)  (gated output)
    ↓
  out_proj → Output (B, L, D)

References:
  - Paper: "Transformers are SSMs" (Dao & Gu, ICML 2024)
  - https://github.com/state-spaces/mamba
"""

import jax
import jax.numpy as jnp
from flax import nnx
import math

from .ssd import ssd_linear_attention_chunked


# =============================================================================
# Helper Functions
# =============================================================================

def elu_plus_one(x: jax.Array) -> jax.Array:
    """Feature map: ELU(x) + 1 to ensure positivity."""
    return jax.nn.elu(x) + 1.0


def softplus(x: jax.Array) -> jax.Array:
    """Softplus activation: log(1 + exp(x)), numerically stable."""
    return jnp.where(x > 20, x, jnp.log1p(jnp.exp(x)))


# =============================================================================
# Mamba2Block: The Core Mamba-2 Layer
# =============================================================================

class Mamba2Block(nnx.Module):
    """
    Mamba-2 Block using State Space Duality (SSD).

    This implements the SSD layer which can be viewed as:
    1. SSM view: h_t = A_t * h_{t-1} + B_t * x_t;  y_t = C_t^T * h_t
    2. Attention view: y = (L ⊙ CB^T) x  where L is the causal decay mask

    The SSD formulation allows efficient chunked computation that leverages
    tensor cores, achieving 2-8x speedup over Mamba-1.

    Key insight: In Mamba2, A is scalar × identity (not diagonal), which enables:
    - Larger state dimension (64-128 vs 16)
    - Multi-head structure for expressivity
    - Efficient matrix operations on tensor cores

    Parameters:
    -----------
    d_model : int
        Input/output model dimension
    d_state : int
        SSM state dimension per head (default: 64)
    d_conv : int
        Short convolution kernel size (default: 4)
    expand : int
        Expansion factor for inner dimension (default: 2)
    headdim : int
        Dimension per head (default: 64)
    ngroups : int
        Number of groups for B, C sharing (default: 1)
    chunk_size : int
        Chunk size for SSD computation (default: 64)
    use_feature_map : bool
        Whether to apply ELU+1 feature map to B, C (default: True)
    A_init_range : tuple
        Range for A initialization (default: (1, 16))
    dt_min, dt_max : float
        Range for dt initialization
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        use_feature_map: bool = True,
        A_init_range: tuple = (1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False,
        conv_bias: bool = True,
        *,
        rngs: nnx.Rngs
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.use_feature_map = use_feature_map

        # Derived dimensions
        self.d_inner = d_model * expand  # Inner/expanded dimension
        assert self.d_inner % headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // headdim

        # Ensure ngroups divides nheads (for grouped B, C)
        assert self.nheads % ngroups == 0, "nheads must be divisible by ngroups"

        # =====================================================================
        # Input Projection: projects to [z, x, B, C, dt]
        # =====================================================================
        # z: gate signal (d_inner)
        # x: main input to SSM (d_inner)
        # B: state matrix (ngroups * d_state) - shared across heads in group
        # C: observation matrix (ngroups * d_state) - shared across heads in group
        # dt: time delta per head (nheads)
        d_in_proj = (
            self.d_inner +          # z (gate)
            self.d_inner +          # x (SSM input)
            ngroups * d_state +     # B
            ngroups * d_state +     # C
            self.nheads             # dt (one per head)
        )
        self.in_proj = nnx.Linear(d_model, d_in_proj, use_bias=bias, rngs=rngs)

        # =====================================================================
        # Short Convolution: local context modeling
        # =====================================================================
        # Depthwise 1D conv on x for short-range dependencies
        # This helps capture local patterns before the SSM processes them
        # Uses 'same' padding equivalent via manual padding for causality
        self.conv_weight = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (d_conv, self.d_inner))
        )
        if conv_bias:
            self.conv_bias = nnx.Param(jnp.zeros(self.d_inner))
        else:
            self.conv_bias = None

        # =====================================================================
        # SSM Parameters
        # =====================================================================
        # A: Decay rate per head (scalar × identity structure in Mamba2)
        # Initialized log-uniformly in [A_init_range[0], A_init_range[1]]
        A_init = jnp.log(
            jnp.linspace(A_init_range[0], A_init_range[1], self.nheads)
        )
        self.A_log = nnx.Param(A_init)

        # dt bias: for converting dt logits to actual dt values
        # Initialized so that softplus(dt_bias) is in [dt_min, dt_max]
        dt_init = jnp.exp(
            jnp.linspace(math.log(dt_min), math.log(dt_max), self.nheads)
        )
        # Inverse softplus to get the bias
        dt_bias_init = dt_init + jnp.log(-jnp.expm1(-dt_init))
        self.dt_bias = nnx.Param(dt_bias_init)

        # D: Skip connection / direct feedthrough (like attention's residual)
        self.D = nnx.Param(jnp.ones(self.nheads))

        # =====================================================================
        # Output: RMSNorm + Gating + Projection
        # =====================================================================
        self.norm = nnx.RMSNorm(self.d_inner, rngs=rngs)
        self.out_proj = nnx.Linear(self.d_inner, d_model, use_bias=bias, rngs=rngs)

    def __call__(
        self,
        u: jax.Array,
        pad_mask: jax.Array | None = None
    ) -> jax.Array:
        """
        Forward pass of Mamba2 block.

        Parameters:
        -----------
        u : jax.Array
            Input tensor of shape (B, L, D)
        pad_mask : jax.Array, optional
            Padding mask of shape (B, L) with 1 for valid, 0 for padding

        Returns:
        --------
        jax.Array
            Output tensor of shape (B, L, D)
        """
        B, L, D = u.shape

        # =================================================================
        # Step 1: Input projection → [z, x, B, C, dt]
        # =================================================================
        zxBCdt = self.in_proj(u)  # (B, L, d_in_proj)

        # Split into components
        d_inner = self.d_inner
        d_state = self.d_state
        ngroups = self.ngroups
        nheads = self.nheads

        z, x, B_proj, C_proj, dt_logits = jnp.split(
            zxBCdt,
            [d_inner, 2 * d_inner, 2 * d_inner + ngroups * d_state,
             2 * d_inner + 2 * ngroups * d_state],
            axis=-1
        )
        # z: (B, L, d_inner) - gate signal
        # x: (B, L, d_inner) - SSM input
        # B_proj: (B, L, ngroups * d_state)
        # C_proj: (B, L, ngroups * d_state)
        # dt_logits: (B, L, nheads)

        # =================================================================
        # Step 2: Short convolution on x (causal, depthwise)
        # =================================================================
        x = self._causal_conv1d(x)  # (B, L, d_inner)
        x = jax.nn.silu(x)  # SiLU activation after conv

        # =================================================================
        # Step 3: Prepare B, C, dt for SSD
        # =================================================================
        # Reshape B, C to (B, L, ngroups, d_state)
        B_mat = B_proj.reshape(B, L, ngroups, d_state)
        C_mat = C_proj.reshape(B, L, ngroups, d_state)

        # Apply feature map if enabled (ensures positivity for linear attention)
        if self.use_feature_map:
            B_mat = elu_plus_one(B_mat)
            C_mat = elu_plus_one(C_mat)

        # Expand B, C to per-head: (B, L, nheads, d_state)
        # Each group is shared across (nheads // ngroups) heads
        heads_per_group = nheads // ngroups
        B_mat = jnp.repeat(B_mat, heads_per_group, axis=2)  # (B, L, nheads, d_state)
        C_mat = jnp.repeat(C_mat, heads_per_group, axis=2)  # (B, L, nheads, d_state)

        # Compute dt: softplus(dt_logits + dt_bias) for stability
        dt = softplus(dt_logits + self.dt_bias.get_value())  # (B, L, nheads)

        # Compute decay a = exp(-dt * A)
        # A is stored as log(A) for numerical stability
        A = jnp.exp(self.A_log.get_value())  # (nheads,)
        a = jnp.exp(-dt * A)  # (B, L, nheads) - decay factors in (0, 1]

        # =================================================================
        # Step 4: Reshape x for multi-head SSD
        # =================================================================
        # x: (B, L, d_inner) → (B, L, nheads, headdim)
        x_heads = x.reshape(B, L, nheads, self.headdim)

        # Transpose for SSD: (B, H, L, D)
        # Our ssd_linear_attention_chunked expects (B, H, N, D)
        q = jnp.transpose(C_mat, (0, 2, 1, 3))  # (B, nheads, L, d_state)
        k = jnp.transpose(B_mat, (0, 2, 1, 3))  # (B, nheads, L, d_state)
        v = jnp.transpose(x_heads, (0, 2, 1, 3))  # (B, nheads, L, headdim)
        a_t = jnp.transpose(a, (0, 2, 1))  # (B, nheads, L)

        # =================================================================
        # Step 5: SSD Core Computation
        # =================================================================
        # y = SSD(q=C, k=B, v=x, a=decay)
        # This is where the magic happens - chunked linear attention with decay
        y = ssd_linear_attention_chunked(
            q, k, v, a_t,
            pad_mask=pad_mask,
            chunk_size=self.chunk_size
        )  # (B, nheads, L, headdim)

        # Add skip connection: y = y + D * x
        # D acts like the residual connection in attention
        D = self.D.get_value()  # (nheads,)
        y = y + D[None, :, None, None] * v  # (B, nheads, L, headdim)

        # Transpose back: (B, nheads, L, headdim) → (B, L, d_inner)
        y = jnp.transpose(y, (0, 2, 1, 3))  # (B, L, nheads, headdim)
        y = y.reshape(B, L, self.d_inner)  # (B, L, d_inner)

        # =================================================================
        # Step 6: Output gating and projection
        # =================================================================
        # RMSNorm + gate with z
        y = self.norm(y)  # RMSNorm
        y = y * jax.nn.silu(z)  # Gating with SiLU(z)

        # Final projection back to model dimension
        out = self.out_proj(y)  # (B, L, d_model)

        return out

    def _causal_conv1d(self, x: jax.Array) -> jax.Array:
        """
        Apply causal 1D depthwise convolution.

        Causal means we only look at past tokens, not future.
        Depthwise means each channel is convolved independently.

        Parameters:
        -----------
        x : jax.Array
            Input of shape (B, L, d_inner)

        Returns:
        --------
        jax.Array
            Output of shape (B, L, d_inner)
        """
        B, L, D = x.shape
        k = self.d_conv

        # Pad left side for causal convolution
        # We pad (k-1) on the left so output[t] only depends on input[t-k+1:t+1]
        x_padded = jnp.pad(x, ((0, 0), (k - 1, 0), (0, 0)))  # (B, L+k-1, D)

        # Use jax.lax.conv_general_dilated for depthwise convolution
        # Input: (B, L+k-1, D) -> transpose to (B, D, L+k-1) for NCHW-like format
        x_t = jnp.transpose(x_padded, (0, 2, 1))  # (B, D, L+k-1)

        # Kernel: (k, D) -> reshape to (D, 1, k) for depthwise conv
        # feature_group_count=D makes it depthwise
        kernel = self.conv_weight.get_value()  # (k, D)
        kernel = jnp.transpose(kernel, (1, 0))  # (D, k)
        kernel = kernel[:, None, :]  # (D, 1, k) - [out_channels, in_channels/groups, width]

        # Depthwise 1D convolution
        # dimension_numbers: (batch, channels, spatial) for input, kernel, output
        y = jax.lax.conv_general_dilated(
            x_t,                          # (B, D, L+k-1)
            kernel,                       # (D, 1, k)
            window_strides=(1,),
            padding='VALID',              # No padding (already padded)
            feature_group_count=D,        # Depthwise: each channel convolved separately
            dimension_numbers=('NCH', 'OIH', 'NCH')  # N=batch, C=channel, H=spatial
        )  # (B, D, L)

        # Transpose back to (B, L, D)
        y = jnp.transpose(y, (0, 2, 1))  # (B, L, D)

        # Add bias if present
        if self.conv_bias is not None:
            y = y + self.conv_bias.get_value()

        return y


# =============================================================================
# Test Functions
# =============================================================================

def test_mamba2_block():
    """
    Test the Mamba2Block implementation.

    Verifies:
    1. Forward pass runs without error
    2. Output shape is correct
    3. Gradients flow properly
    """
    print("=" * 60)
    print("Testing Mamba2Block")
    print("=" * 60)

    # Test configuration
    batch_size = 2
    seq_len = 128
    d_model = 256
    d_state = 64
    headdim = 64
    expand = 2
    chunk_size = 32

    # Create model
    rngs = nnx.Rngs(0)
    model = Mamba2Block(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        expand=expand,
        headdim=headdim,
        ngroups=1,
        chunk_size=chunk_size,
        use_feature_map=True,
        rngs=rngs
    )

    # Create random input
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, seq_len, d_model))

    # Forward pass
    print(f"\nInput shape: {x.shape}")
    print(f"d_model={d_model}, d_state={d_state}, expand={expand}")
    print(f"d_inner={model.d_inner}, nheads={model.nheads}, headdim={headdim}")

    y = model(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {y.shape}"
    print("✓ Output shape correct")

    # Test with padding mask
    pad_mask = jnp.ones((batch_size, seq_len))
    pad_mask = pad_mask.at[:, -10:].set(0)  # Last 10 tokens are padding
    y_masked = model(x, pad_mask=pad_mask)
    print(f"Output with mask shape: {y_masked.shape}")
    assert y_masked.shape == (batch_size, seq_len, d_model)
    print("✓ Masked forward pass works")

    # Test gradient flow
    def loss_fn(model, x):
        y = model(x)
        return jnp.mean(y ** 2)

    grads = nnx.grad(loss_fn)(model, x)
    print("✓ Gradients computed successfully")

    # Check some parameter gradients exist
    in_proj_grad = grads.in_proj.kernel.get_value()
    print(f"in_proj gradient shape: {in_proj_grad.shape}, mean: {jnp.mean(jnp.abs(in_proj_grad)):.6f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_mamba2_different_configs():
    """Test Mamba2Block with different configurations."""
    print("\n" + "=" * 60)
    print("Testing different configurations")
    print("=" * 60)

    configs = [
        {"d_model": 128, "d_state": 32, "headdim": 32, "expand": 2, "ngroups": 1},
        {"d_model": 256, "d_state": 64, "headdim": 64, "expand": 2, "ngroups": 2},
        {"d_model": 512, "d_state": 128, "headdim": 64, "expand": 2, "ngroups": 4},
    ]

    for i, config in enumerate(configs):
        print(f"\nConfig {i + 1}: {config}")

        rngs = nnx.Rngs(i)
        model = Mamba2Block(
            d_model=config["d_model"],
            d_state=config["d_state"],
            headdim=config["headdim"],
            expand=config["expand"],
            ngroups=config["ngroups"],
            chunk_size=32,
            rngs=rngs
        )

        key = jax.random.PRNGKey(i)
        x = jax.random.normal(key, (2, 64, config["d_model"]))
        y = model(x)

        print(f"  Input: {x.shape} → Output: {y.shape}")
        print(f"  nheads={model.nheads}, d_inner={model.d_inner}")
        assert y.shape == x.shape
        print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All configuration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_mamba2_block()
    test_mamba2_different_configs()

