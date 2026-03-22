import gzip
import logging
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from kimi_gated_delta_attn.naive.kimi_gated_delta_attn import KimiDeltaAttention, _kda_gate

LOGGER = logging.getLogger(__name__)
_SPARK_BARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return ""
    peak = float(values.max())
    if peak <= 0:
        return _SPARK_BARS[0] * values.size
    scaled = np.clip(np.rint((len(_SPARK_BARS) - 1) * values / peak).astype(int), 0, len(_SPARK_BARS) - 1)
    return "".join(_SPARK_BARS[index] for index in scaled)


def _log_tensor_summary(name: str, tensor: jax.Array) -> None:
    array = np.asarray(jax.device_get(tensor))
    flat = array.astype(np.float32).reshape(-1)
    hist, _ = np.histogram(flat, bins=min(8, max(2, flat.size)))
    LOGGER.info(
        "📊 %-18s shape=%s dtype=%s min=% .5f max=% .5f mean=% .5f std=% .5f hist=%s",
        name,
        array.shape,
        array.dtype,
        float(flat.min()),
        float(flat.max()),
        float(flat.mean()),
        float(flat.std()),
        _sparkline(hist),
    )


def _find_trace_annotations(trace_file: Path, *needles: str) -> set[str]:
    remaining = set(needles)
    found: set[str] = set()
    with gzip.open(trace_file, "rt", encoding="utf-8", errors="ignore") as handle:
        for chunk in iter(lambda: handle.read(1 << 15), ""):
            for needle in tuple(remaining):
                if needle in chunk:
                    found.add(needle)
                    remaining.remove(needle)
            if not remaining:
                break
    return found


def _profiler_log_dir() -> Path:
    configured = os.environ.get("KIMI_PROFILE_DIR")
    if configured:
        path = Path(configured).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="kimi-prof-"))


def _latest_trace_dir(log_dir: Path) -> Path:
    trace_root = log_dir / "plugins" / "profile"
    trace_dirs = sorted(trace_root.iterdir(), key=lambda path: path.stat().st_mtime)
    assert trace_dirs, f"Expected profiler output under {trace_root}."
    return trace_dirs[-1]


def _perfetto_viewer_url(trace_name: str, port: int = 9001) -> str:
    return f"https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/{trace_name}"


def _new_model(seed: int = 0) -> KimiDeltaAttention:
    return KimiDeltaAttention(
        hidden_size=16,
        conv_size=3,
        head_dim=4,
        num_heads=2,
        head_k_dim=4,
        num_k_heads=2,
        layer_idx=0,
        rngs=nnx.Rngs(seed),
    )


def test_masked_forward_emits_default_jax_profiler_trace() -> None:
    model = _new_model(seed=0)
    hidden_states = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 16))
    attention_mask = jnp.array(
        [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=jnp.float32,
    )

    _log_tensor_summary("hidden_states", hidden_states)
    _ = jax.block_until_ready(model(hidden_states, attention_mask=attention_mask))

    log_dir = _profiler_log_dir()
    create_perfetto_link = os.environ.get("KIMI_CREATE_PERFETTO_LINK") == "1"
    with jax.profiler.trace(
        log_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=True,
    ):
        with jax.profiler.TraceAnnotation("kimi_naive_forward"):
            for step_num in (1, 2):
                with jax.profiler.StepTraceAnnotation("naive_forward_step", step_num=step_num):
                    output = model(hidden_states, attention_mask=attention_mask)
                    output = jax.block_until_ready(output)

    trace_dir = _latest_trace_dir(log_dir)
    trace_files = sorted(trace_dir.glob("*.trace.json.gz"))
    assert trace_files, f"Expected a default JAX profiler trace under {trace_dir}."

    trace_file = trace_files[0]
    perfetto_trace = trace_dir / "perfetto_trace.json.gz"
    assert perfetto_trace.exists(), f"Expected Perfetto trace under {trace_dir}."

    found = _find_trace_annotations(trace_file, "kimi_naive_forward", "naive_forward_step")
    LOGGER.info(
        "🧭 profiler dir=%s trace=%s perfetto=%s annotations=%s",
        trace_dir,
        trace_file,
        perfetto_trace,
        sorted(found),
    )
    LOGGER.info(
        "🔗 perfetto viewer: upload %s to https://ui.perfetto.dev or serve %s and open %s (inspect step 2 for the steadiest forward-pass timing)",
        perfetto_trace,
        trace_dir,
        _perfetto_viewer_url(perfetto_trace.name),
    )

    assert found == {"kimi_naive_forward", "naive_forward_step"}

    _log_tensor_summary("masked_output", output)
    assert output.shape == hidden_states.shape
    np.testing.assert_allclose(np.asarray(output[1, 4:]), 0.0, atol=1e-6)


def test_attention_mask_4d_matches_2d_mask() -> None:
    model = _new_model(seed=3)
    hidden_states = jax.random.normal(jax.random.PRNGKey(7), (2, 8, 16))
    mask_2d = jnp.array(
        [[1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=jnp.float32,
    )
    mask_4d = mask_2d[:, None, :, None]

    output_2d = jax.block_until_ready(model(hidden_states, attention_mask=mask_2d))
    output_4d = jax.block_until_ready(model(hidden_states, attention_mask=mask_4d))

    _log_tensor_summary("output_2d_mask", output_2d)
    _log_tensor_summary("output_4d_mask", output_4d)

    np.testing.assert_allclose(output_2d, output_4d, atol=1e-6, rtol=1e-6)


def test_chunk_mode_matches_recurrent_mode_for_long_sequences() -> None:
    model = _new_model(seed=5)
    hidden_states = jax.random.normal(jax.random.PRNGKey(11), (1, 65, 16))
    attention_mask = jnp.concatenate([jnp.ones((1, 60), dtype=jnp.float32), jnp.zeros((1, 5), dtype=jnp.float32)], axis=1)

    model.mode = "fused_recurrent"
    recurrent_output = jax.block_until_ready(model(hidden_states, attention_mask=attention_mask))

    model.mode = "chunk"
    chunk_output = jax.block_until_ready(model(hidden_states, attention_mask=attention_mask))

    delta = np.abs(np.asarray(recurrent_output - chunk_output))
    LOGGER.info(
        "🔍 recurrent-vs-chunk max=% .6f mean=% .6f diff=%s",
        float(delta.max()),
        float(delta.mean()),
        _sparkline(delta.reshape(-1)[:8] if delta.size >= 8 else np.pad(delta.reshape(-1), (0, 8 - delta.size))),
    )

    np.testing.assert_allclose(chunk_output, recurrent_output, atol=1e-6, rtol=1e-6)


def test_kda_gate_handles_projected_inputs_and_bias_reshaping() -> None:
    batch_size, seq_len, num_heads, head_dim = 2, 3, 2, 4
    projected = jnp.linspace(-1.0, 1.0, batch_size * seq_len * num_heads * head_dim, dtype=jnp.float32)
    projected = projected.reshape(batch_size, seq_len, num_heads * head_dim)
    explicit = projected.reshape(batch_size, seq_len, num_heads, head_dim)
    A_log = jnp.log(jnp.array([[[[2.0], [0.5]]]], dtype=jnp.float32))
    g_bias = jnp.linspace(-0.25, 0.25, num_heads * head_dim, dtype=jnp.float32)

    projected_gate = _kda_gate(projected, A_log, head_dim=head_dim, g_bias=g_bias)
    explicit_gate = _kda_gate(explicit, A_log, head_dim=head_dim, g_bias=g_bias.reshape(num_heads, head_dim))

    _log_tensor_summary("projected_gate", projected_gate)

    assert projected_gate.shape == (batch_size, seq_len, num_heads, head_dim)
    np.testing.assert_allclose(projected_gate, explicit_gate, atol=1e-6, rtol=1e-6)
    assert bool(jnp.all((projected_gate > 0.0) & (projected_gate <= 1.0)))
