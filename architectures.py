import jax
from jax.experimental import stax
from jax.nn import leaky_relu

from utils import DenseNoBias


def fully_connected(units, classes, activation, init, bias=False):
    """
    Implements a simple fully-connected neural network using the library stax based on jax.
    :param units:       list, list of hidden layer sizes, excluding output
    :param classes:     int, number of classes, i.e. outputs of the network
    :param activation:  str, non-linearity , one of 'linear', 'relu', 'tanh', 'elu', 'sigmoid', 'leaky_relu'
    :param init:        str, initialization scheme, one of 'orthogonal', 'uniform', 'glorot'
    :param bias:        bool, use bias in layers or not

    :return:    init_fn, apply_fn
    """
    if activation is None or activation == 'linear':
        if bias == False:
            architecture = [DenseNoBias(i, W_init=init()) for i in units]
            if classes == 2:
                # If only two classes, we use one output and encode labels as -1,1
                architecture += [DenseNoBias(1, W_init=init())]
            else:
                architecture += [DenseNoBias(classes, W_init=init())]
        else:
            architecture = [stax.Dense(i, W_init=init()) for i in units]
            if classes == 2:
                architecture += [stax.Dense(1, W_init=init())]
            else:
                architecture += [stax.Dense(classes, W_init=init())]

    elif activation == 'relu':
        architecture = []
        for i in range(len(units)):
            architecture += [DenseNoBias(units[i], W_init=init()), stax.Relu]
        if classes == 2:
            architecture += [DenseNoBias(1)]
        else:
            architecture += [DenseNoBias(classes)]

    elif activation == 'tanh':
        architecture = []
        for i in range(len(units)):
            architecture += [DenseNoBias(units[i], W_init=init()), stax.Tanh]
        architecture += [DenseNoBias(classes)]

    elif activation == 'elu':
        architecture = []
        for i in range(len(units)):
            architecture += [DenseNoBias(units[i], W_init=init()), stax.Elu]
        architecture += [DenseNoBias(classes)]

    elif activation == 'sigmoid':
        architecture = []
        for i in range(len(units)):
            architecture += [DenseNoBias(units[i], W_init=init()), stax.Sigmoid]
        architecture += [DenseNoBias(classes)]

    elif activation == 'leaky_relu':
        def leaky_relu_fixed(x):
            return leaky_relu(x, negative_slope=0.01)

        architecture = []
        for i in range(len(units)):
            architecture += [DenseNoBias(units[i]), jax.experimental.stax.elementwise(leaky_relu_fixed)]
        if classes == 2:
            architecture += [DenseNoBias(1)]
        else:
            architecture += [DenseNoBias(classes)]

    init_fn, apply_fn = stax.serial(
        *architecture
    )

    return init_fn, apply_fn
