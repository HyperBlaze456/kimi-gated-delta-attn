# Task identification


## 0. Understand linear attention
From-scratch implementation of linear attention

Do some tweaks, try to make eager implementation using flax NNX more 'not eager'. Kernel level optimization won't be here,
however additional consideration like reusing the state might be here.

Maybe start chunking already?


## 1. Naively implement gated delta attention
Implement naively using basic Flax modules and jax.numpy based computations.

