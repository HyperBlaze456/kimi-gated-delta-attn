import jax
import jax.numpy as jnp
from flax import nnx


def elu_plus_one(x: jax.Array) -> jax.Array:
    # uses this positive lock funciton instead of softmax
    return jax.nn.elu(x) + 1.0

