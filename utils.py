import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, ones, zeros

from data import *


def DenseNoBias(out_dim, W_init=glorot_normal()):
    """Stax doesn't offer layers that disable the use of bias. We provide the neccesary layer here.
    Layer constructor function for a dense (fully-connected) layer without using bias."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W = W_init(k1, (input_shape[-1], out_dim))
        return output_shape, (W)

    def apply_fun(params, inputs, **kwargs):
        W = params
        return jnp.dot(inputs, W)

    return init_fun, apply_fun


def flatten(v):
    """Takes parameters of NN in the form of a pytree and maps it to its vector representation"""
    def f(v):
        leaves, _ = jax.tree_util.tree_flatten(v)
        return jnp.concatenate([x.ravel(order='F') for x in leaves])
    out, pullback = jax.vjp(f, v)
    return out, lambda x: pullback(x)[0]
