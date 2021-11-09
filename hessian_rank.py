import jax.numpy as jnp
from jax import *
from jax.config import config
from jax.experimental.stax import softmax, logsoftmax
from initializers import get_init

from data import get_dataset
from hessians import outer_prod, loss_hessian
from architectures import fully_connected

from dataloader import DatasetTorch
from torch.utils.data import DataLoader


config.update("jax_enable_x64", True)

# Define the hyperparameters
n_train = 50                                                 # Sample size
dim = 25                                                     # Dimension of data
widths = [5, 10]                                             # Width of the network, excluding last layer
classes = 10                                                 # Number of classes
bs = 10

all_widths = [dim] + widths + [classes]
p = sum([all_widths[i] * all_widths[i + 1] for i in range(len(all_widths)-1)])

# Initialize seed
key = random.PRNGKey(1)

# Define the data, we will choose down-scaled MNIST
data = get_dataset('MNIST', n_train=n_train, n_test=1, dim=dim, classes=classes)

# Define a train loader so that we can batch the Hessian calculation
train_loader = DataLoader(DatasetTorch(data.x_train, data.y_train), batch_size=bs)

# Choose initialization
init = get_init('glorot')

# Define linear neural network architecture
init_fn, apply_fn = fully_connected(units=widths, classes=classes, activation='linear', init=init)

# Initialize the parameters
_, params = init_fn(key, (-1, dim))

# Make sure parameters are double precision
params = [jnp.double(param) for param in params]

# Define the loss function, you can choose from 'mse', 'cross' and 'cosh'
loss_name = 'cross'

if loss_name == 'mse':
    cross = False


    def loss(preds, targets):
        return 1 / 2 * jnp.sum((preds - targets) ** 2)


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return 1 / 2 * jnp.sum((preds - targets) ** 2)

if loss_name == 'cross':
    cross = True


    def loss(preds, targets):
        return -jnp.sum(logsoftmax(preds) * targets)


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return -jnp.sum(logsoftmax(preds) * targets)

if loss_name == 'cosh':
    cross = False


    def loss(preds, targets):
        return jnp.sum(jnp.log(jnp.cosh(preds - targets)))


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return loss(preds, targets)


H_L, H_outer = jnp.zeros(shape=(p, p)), jnp.zeros(shape=(p, p))
cov = jnp.zeros(shape=(dim, dim))

for batch_input, batch_label in train_loader:
    batch_input, batch_label = (batch_input.numpy(), batch_label.numpy())
    # Calculate the covariance
    cov += batch_input.T @ batch_input
    # Calculate loss hessian
    H_L += loss_hessian(loss_params, params, batch_input, batch_label)
    # Calculate the outer gradient product
    H_outer += outer_prod(loss, apply_fn, params, batch_input, batch_label, cross=cross)

# To save time we calculate the functional Hessian as the difference
H_F = H_L - H_outer

# Calculate the ranks
rank_cov = jnp.linalg.matrix_rank(cov)
rank_L = jnp.linalg.matrix_rank(H_L)
rank_outer = jnp.linalg.matrix_rank(H_outer)
rank_F = jnp.linalg.matrix_rank(H_F)

# For cross entropy we have to slightly adapt the formula, check out the paper for more details
if loss_name == 'cross':
    classes = classes - 1

s = jnp.min(jnp.array([rank_cov, classes]))
q = jnp.min(jnp.array([rank_cov, classes] + widths))

pred_F = 2 * q * sum(widths) + 2 * q * s - (len(widths) + 1) * q ** 2
pred_outer = (rank_cov + classes - q) * q
pred_L = pred_F + pred_outer + q * (q - 2 * s)

print('Rank of Functional Hessian is ' + str(rank_F) + ' and the prediction is ' + str(pred_F))

print('Rank of Gradient Outer Product is ' + str(rank_outer) + ' and the prediction is ' + str(pred_outer))

print('Rank of Loss Hessian is ' + str(rank_L) + ' and the prediction is ' + str(pred_L))