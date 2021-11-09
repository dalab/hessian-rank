import jax
import jax.numpy as np
from jax import jit
from jax import jvp, grad, jit, jacfwd, vmap
from jax.experimental.stax import softmax
import jax.numpy as jnp
from utils import flatten


def outer_prod(loss, apply_fn, params, inputs, targets, cross=False):
    """
    Implements the outer product calculation, weighted by residuals coming from the loss

    :param loss:        function f(x, y), implementing the loss as a function of inputs and targets
    :param apply_fn:    function f(p, x), implementing the network as a function of params and inputs
    :param params:      pytree, containing parameters of the model
    :param inputs:      jnp.array, inputs to the model
    :param targets:     jnp.array, targets
    :param cross:       bool, True if loss is the cross entropy loss

    :return:            jnp.array, outer product term
    """
    # Flatten pytree to vector representation
    flat_params, unflatten_params = flatten(params)
    # Calculate prediction of models
    preds = apply_fn(params, inputs)

    if cross:
        # In case of cross entropy, we calculate second order term from loss "by hand"
        beta = softmax(preds)
    else:
        # Otherwise we just calculate the hessian of the loss
        beta = jnp.diagonal(vmap(jax.hessian(lambda z, t: loss(z, t)), in_axes=0)(preds, targets), axis1=1, axis2=2)

    beta = jnp.expand_dims(beta, axis=2)

    def f_n_flat(flat_params):
        return apply_fn(unflatten_params(flat_params), inputs)

    # Calculate the jacobian of the model
    jac = jacfwd(f_n_flat)(flat_params)
    # Form outer product and weight by the residuals beta, summing over the class and sample axes
    res = jnp.tensordot(jac, jac * beta, axes=[[0, 1], [0, 1]])

    # Calculate the cross term, 0 for all losses except cross entropy since softmax "connects" the summands and causes
    # non-vanishing mixed derivatives
    cross_term = 0
    if cross:
        cross_term = jnp.sum(beta * jac, axis=1)
        cross_term = cross_term.T @ cross_term

    return res - cross_term


def functional_hessian(loss, apply_fn, params, inputs, targets):
    """
    Calculate functional hessian part, usually the most memory-intensive so we usually avoid it by calculating
    H_F = H_L - outer

    :param loss:            function f(x, y), implementing the loss as a function of inputs and targets
    :param apply_fn:        function f(p, x), implementing the network as a function of params and inputs
    :param params:          pytree, containing parameters of the model
    :param inputs:          jnp.array, inputs to the model
    :param targets:         jnp.array, targets

    :return:                jnp.array, functional hessian term
    """
    def f_n(params):
        return apply_fn(params, inputs)

    # Flatten the parameters
    flat_params, unflatten_params = flatten(params)

    # Calculate first derivatives
    preds = apply_fn(params, inputs)
    if loss == 'cross':
        alpha = -(1 - softmax(preds, axis=1)) * targets
    else:
        alpha = vmap(grad(lambda z, t: loss(z, t)), in_axes=0)(preds, targets)

    # Calculate Hessian of the function mapping (size K x P x P )
    H_F_all = jax.hessian(lambda t: f_n(unflatten_params(t)))(flat_params)

    # Number of outputs of the network
    K = H_F_all.shape[1]

    # Calculate the sum of all component Hessians (size P x P )
    if K == 1:
        H_F = np.expand_dims(alpha, axis=(2, 3)) * H_F_all
        H_F = np.sum(H_F, axis=(0, 1)).squeeze()
    else:
        res = np.expand_dims(alpha, axis=(2, 3))
        H_F = np.sum(res * H_F_all, axis=(0, 1))

    return H_F


def loss_hessian(loss, params, inputs, targets):
    """
    Implements calculation of the loss hessian
    :param loss:        function f(p, x, y), implementing the loss as a function of params, inputs, targets
    :param params:      pytree, containing parameters of the model
    :param inputs:      jnp.array, inputs to the model
    :param targets:     jnp.array, targets

    :return:            jnp.array, Loss hessian term
    """
    # Flatten the parameters into a vector
    flat_params, unflatten_params = flatten(params)
    # Define the loss only as a function of the parameters, keeping inputs and targets fixed
    loss_params = lambda p: loss(p, inputs, targets)

    # Calculate Hessian of the loss (size P x P )
    H_L = jax.hessian(lambda t: loss_params(unflatten_params(t)))(flat_params)

    return H_L
