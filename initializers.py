from jax.nn.initializers import glorot_normal, normal, ones, zeros, uniform, he_uniform
from jax import random
import jax.numpy as jnp


def orthogonal_init():
    """Implements orthogonal initialization, i.e. sampling w.r.t. Haar measure over the space of orthogonal matrices"""
    def init(key, shape):
        W = 1 / shape[0] * random.normal(key, shape)
        if shape[0] < shape[1]:
            Q, _ = jnp.linalg.qr(W.T)
            return Q.T
        else:
            Q, _ = jnp.linalg.qr(W)
            return Q

    return init


def uniform_init():
    """Uniform initialization"""
    def init(key, shape):
        W = 1 / shape[0] * random.uniform(key, shape)

        return W

    return init


def get_init(name):
    """Helper function returning the desired initialization scheme"""
    if name == 'orthogonal':
        return orthogonal_init
    if name == 'uniform':
        return uniform_init
    if name == 'glorot':
        return glorot_normal