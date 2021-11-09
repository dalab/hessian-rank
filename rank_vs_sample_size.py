import os

from jax import *
from jax.config import config
import pandas as pd

from hessians import outer_prod, loss_hessian
from architectures import fully_connected
from dataloader import DatasetTorch
from torch.utils.data import DataLoader
from jax.experimental.stax import softmax, logsoftmax
from initializers import *
import argparse
from data import get_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--units', default='10,10', type=str)
parser.add_argument('--loss', default='mse', type=str)
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--init', default='glorot', type=str)
parser.add_argument('--dim', default=64, type=int)
parser.add_argument('--K', default=10, type=int)

args = parser.parse_args()


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = ''

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
key = random.PRNGKey(1)

n_train_max = 400

m = args.units.split(',')
depth = len(m)
K = args.K
m = [int(m[i]) for i in range(depth)]
d = args.dim
in_d = args.dim

init = get_init(args.init)

unit_string = str(d) + 'x'
for i in range(depth):
    unit_string += str(m[i]) + 'x'
unit_string += str(K)

bs = args.batch_size
iters = n_train_max // bs

n_trains = [bs + i * bs for i in range(iters)]

# Prepare dictionaries to store results
preds = {'rank_L': [], 'rank_F': [], 'rank_outer': []}
ranks = {'rank_L': [], 'rank_F': [], 'rank_outer': []}
ranks_cov = []

dir_path = os.path.dirname(os.path.realpath(__file__))
store_path = dir_path + '/results/store/samplesize/'
store_path += args.loss + '/' + args.dataset + '/' + unit_string + '/'
temp_path = dir_path + '/results/temporary/samplesize/'
temp_path += args.loss + '/' + args.dataset + '/' + unit_string + '/'

try:
    # Create directory to store results
    os.mkdir(store_path)
    print("Directory ", store_path,  " Created ")
except FileExistsError:
    print("Directory ", store_path,  " already exists")

try:
    # Create directory to store intermediate results
    os.mkdir(temp_path)
    print("Directory ", temp_path,  " Created ")
except FileExistsError:
    print("Directory ", temp_path,  " already exists")

# Set parameters in case we don't use cross entropy
K_form = K
cross = False

# Choose loss function and adapt parameters in case of cross entropy
if args.loss == 'mse':
    def loss(preds, targets):
        return 1 / 2 * jnp.sum((preds - targets) ** 2)


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return 1 / 2 * jnp.sum((preds - targets) ** 2)

if args.loss == 'cosh':
    def loss(preds, targets):
        return jnp.sum(jnp.log(jnp.cosh(preds - targets)))


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return loss(preds, targets)


if args.loss == 'cross':
    def loss(preds, targets):
        return -jnp.sum(logsoftmax(preds) * targets)


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return -jnp.sum(logsoftmax(preds) * targets)

    K_form = K - 1
    cross = True

# Define neural architecture
init_fn, apply_fn = fully_connected(units=m, classes=K, activation='linear', init=init)

# Initialize the parameters
_, params = init_fn(key, (-1, in_d))

# Make sure that parameters are float64, calculations are not precise enough otherwise
params = [jnp.double(param) for param in params]

if K == 1:
    # If we only have one output, we use two classes
    data = get_dataset(args.dataset, n_train=n_train_max, n_test=2, dim=d, classes=2)
else:
    data = get_dataset(args.dataset, n_train=n_train_max, n_test=2, dim=d, classes=K)

# Define the trainloader to perform computation in batches
train_loader = DataLoader(DatasetTorch(data.x_train, data.y_train), batch_size=bs, shuffle=False)


cov = jnp.zeros(shape=(in_d, in_d))

# Initialize the functions to compute the Hessians


def loss_hessian_input(batch): return loss_hessian(loss_params, params, batch[0], batch[1])


# Jit them for faster, repeated calculations
loss_hessian_input_jitted = jit(loss_hessian_input)


def outer_hessian_input(batch): return outer_prod(loss, apply_fn, params, batch[0], batch[1],  cross=cross)


outer_hessian_input_jitted = jit(outer_hessian_input)


i = 0
for batch_input, batch_label in train_loader:
    batch = (batch_input.numpy(), batch_label.numpy())
    H_L_batched = loss_hessian_input_jitted(batch)

    if i == 0:
        H_L = H_L_batched
    else:
        H_L = jnp.load(temp_path + 'H_L.npy')
        H_L += H_L_batched

    # Calculate the rank
    rank_L = jnp.linalg.matrix_rank(H_L)
    ranks['rank_L'].append(rank_L)

    # Store it temporarily
    jnp.save(temp_path + 'H_L', H_L)
    # Free memory
    del H_L
    # Calculate outer product
    outer_batched = outer_hessian_input_jitted(batch)
    if i == 0:
        outer = outer_batched
    else:
        outer = jnp.load(temp_path + 'outer.npy')
        outer += outer_batched

    # Calculate the rank
    rank_outer = jnp.linalg.matrix_rank(outer)
    ranks['rank_outer'].append(rank_outer)

    # Calculate the functional hessian, call it H_F to do in-place assignments
    H_F = jnp.load(temp_path + 'H_L.npy')
    H_F = H_F - outer

    # Store it temporarily
    jnp.save(temp_path + 'outer', outer)
    del outer

    # Calculate the rank
    rank_F = jnp.linalg.matrix_rank(H_F)
    ranks['rank_F'].append(rank_F)

    # Update the covariance matrix
    cov_new = batch_input.T @ batch_input
    cov += cov_new.numpy()

    # Calculate the rank
    rank_cov = jnp.linalg.matrix_rank(cov)
    ranks_cov.append(rank_cov)

    # Calculate the corresponding formulas from the paper
    q_rk = jnp.min(jnp.array([rank_cov, K_form]))
    q_all = jnp.min(jnp.array([rank_cov, K_form] + m))
    pred_F = 2 * q_all * sum(m) + 2 * q_all * q_rk - (len(m)+1) * q_all**2
    pred_outer = (rank_cov + K_form - q_all) * q_all
    pred_L = pred_F + pred_outer + q_all * (q_all - 2 * q_rk)

    preds['rank_L'].append(pred_L)
    preds['rank_F'].append(pred_F)
    preds['rank_outer'].append(pred_outer)

    print('Iteration ' + str(i) + ' out of ' + str(iters))
    i += 1

# Save the ranks to the directory
rank_L_frame = pd.DataFrame({'n':  n_trains, 'Rank': jnp.array(ranks['rank_L'])}, dtype=float)
rank_L_frame.to_pickle(path=store_path + 'rank_L')

preds_F_frame = pd.DataFrame({'n': n_trains, 'Pred': jnp.array(preds['rank_F'])}, dtype=float)
preds_F_frame.to_pickle(path=store_path + 'pred_F')

preds_L_frame = pd.DataFrame({'n': n_trains, 'Pred': jnp.array(preds['rank_L'])}, dtype=float)
preds_L_frame.to_pickle(path=store_path + 'pred_L')

preds_outer_frame = pd.DataFrame({'n': n_trains, 'Pred': jnp.array(preds['rank_outer'])}, dtype=float)
preds_outer_frame.to_pickle(path=store_path + 'pred_outer')

rank_F_frame = pd.DataFrame({'n': n_trains, 'Rank': jnp.array(ranks['rank_F'])}, dtype=float)
rank_F_frame.to_pickle(path=store_path + 'rank_F')

rank_outer_frame = pd.DataFrame({'n': n_trains, 'Rank': jnp.array(ranks['rank_outer'])}, dtype=float)
rank_outer_frame.to_pickle(path=store_path + 'rank_outer')