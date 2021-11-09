import os

from jax.config import config
import pandas as pd
from data import get_dataset
from hessians import outer_prod, loss_hessian
from architectures import fully_connected
from jax.experimental.stax import logsoftmax
from dataloader import DatasetTorch
from torch.utils.data import DataLoader
from initializers import *
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = ''

parser = argparse.ArgumentParser()

parser.add_argument('--loss', default='mse', type=str)
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--width', default=25, type=int)
parser.add_argument('--init', default='glorot', type=str)
parser.add_argument('--dim', default=16, type=int)
parser.add_argument('--K', default=10, type=int)

args = parser.parse_args()
config.update("jax_enable_x64", True)
key = random.PRNGKey(1)

# Fix parameters
n_train = 50
bs = args.batch_size
depths_max = 10
width = args.width

K = args.K
d = args.dim
in_d = args.dim

# Choose initializer
init = get_init(args.init)

# Prepare dictionaries to store results
preds = {'rank_L': [], 'rank_F': [], 'rank_outer': []}
ranks = {'rank_L': [], 'rank_F': [], 'rank_outer': []}
ranks_cov = []

# Fix paths to store results
dir_path = os.path.dirname(os.path.realpath(__file__))
store_path = dir_path + '/results/store/depth/'
store_path += args.loss + '/' + args.dataset + '/' + str(width) + '/'

# Fix path to store intermediate results
temp_path = dir_path + '/results/temporary/depth/'
temp_path += args.loss + '/' + args.dataset + '/' + str(width) + '/'

# Create directories
try:
    os.mkdir(store_path)
    print("Directory ", store_path,  " Created ")
except FileExistsError:
    print("Directory ", store_path,  " already exists")

try:
    os.mkdir(temp_path)
    print("Directory ", temp_path,  " Created ")
except FileExistsError:
    print("Directory ", temp_path,  " already exists")

# Initialize loss function along with parameters for the formulas
if args.loss == 'mse':
    def loss(preds, targets):
        return 1 / 2 * jnp.sum((preds - targets) ** 2)


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return 1 / 2 * jnp.sum((preds - targets) ** 2)

    K_form = K
    cross = False

if args.loss == 'cosh':
    def loss(preds, targets):
        return jnp.sum(jnp.log(jnp.cosh(preds - targets)))


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return loss(preds, targets)

    K_form = K
    cross = False

if args.loss == 'cross':
    def loss(preds, targets):
        return -jnp.sum(logsoftmax(preds) * targets)


    def loss_params(params, inputs, targets):
        preds = apply_fn(params, inputs)

        return -jnp.sum(logsoftmax(preds) * targets)

    K_form = K - 1
    cross = True

num_params = []
counter = 0

data_key, key = random.split(key, 2)
if K == 1:
    # If we only have one output, we use two classes
    data = get_dataset(args.dataset, n_train=n_train, n_test=2, dim=d, classes=2)
else:
    data = get_dataset(args.dataset, n_train=n_train, n_test=2, dim=d, classes=K)

train_loader = DataLoader(DatasetTorch(data.x_train, data.y_train), batch_size=bs, shuffle=False)

for depth in range(1, depths_max):
    # Define neural architecture
    m = [width for _ in range(depth)]
    init_fn, apply_fn = fully_connected(units=m, classes=K, activation='linear', init=init)

    # Initialize the parameters
    _, params = init_fn(key, (-1, d))

    # Make sure that parameters are float64, calculations are not precise enough otherwise
    params = [jnp.double(param) for param in params]
    p = sum([param.shape[0] * param.shape[1] for param in params if param.shape[0] != 0])
    num_params.append(p)

    # Create train loader to perform batched computations
    train_loader = DataLoader(DatasetTorch(data.x_train, data.y_train), batch_size=bs)

    # Initialize the Hessians
    H_L, H_F1, H_F, outer = jnp.zeros(shape=(p, p)), jnp.zeros(shape=(p, p)), jnp.zeros(shape=(p, p)), \
                            jnp.zeros(shape=(p, p))

    # Calculate loss hessian
    for batch_input, batch_label in train_loader:
        H_L += loss_hessian(loss_params, params, batch_input.numpy(), batch_label.numpy())

    # Calculate the rank
    rank_L = jnp.linalg.matrix_rank(H_L)
    # Store it temporarily
    jnp.save(temp_path + 'H_L', H_L)
    # Free memory
    del H_L

    # Calculate the outer hessian
    for batch_input, batch_label in train_loader:
        outer += outer_prod(loss, apply_fn, params, batch_input.numpy(), batch_label.numpy(), cross=cross)

    # Calculate the rank
    rank_outer = jnp.linalg.matrix_rank(outer)

    # Calculate the functional hessian
    H_F = jnp.load(temp_path + 'H_L.npy')
    H_F -= outer
    # Free memory
    del outer

    # Calculate the rank
    rank_F = jnp.linalg.matrix_rank(H_F)
    # Free memory
    del H_F

    # Calculate the covariance
    cov, _ = data.get_emp_cov()
    rank_cov = jnp.linalg.matrix_rank(cov)
    ranks_cov.append(rank_cov)

    ranks['rank_L'].append(rank_L)
    ranks['rank_F'].append(rank_F)
    ranks['rank_outer'].append(rank_outer)

    # Calculate predictions for the formula
    q_rk = jnp.min(jnp.array([rank_cov, K_form]))
    q_all = jnp.min(jnp.array([rank_cov, K_form] + m))

    pred_F = 2 * q_all * sum(m) + 2 * q_all * q_rk - (len(m)+1) * q_all**2
    pred_outer = (rank_cov + K_form - q_all) * q_all
    pred_L = pred_F + pred_outer + q_all * (q_all - 2 * q_rk)

    preds['rank_L'].append(pred_L)
    preds['rank_F'].append(pred_F)
    preds['rank_outer'].append(pred_outer)

    print('Iteration ' + str(counter) + ' out of ' + str(depths_max))
    counter += 1
    print(counter)

# Store the results in the folder
depths = [i for i in range(1, depths_max)]
rank_F_frame = pd.DataFrame({'depth': depths, 'Rank': jnp.array(ranks['rank_F'])}, dtype=float)
rank_F_frame.to_pickle(path=store_path + 'rank_F')
rank_L_frame = pd.DataFrame({'depth': depths, 'Rank': jnp.array(ranks['rank_L'])}, dtype=float)
rank_L_frame.to_pickle(path=store_path + 'rank_L')
rank_outer_frame = pd.DataFrame({'depth': depths, 'Rank': jnp.array(ranks['rank_outer'])}, dtype=float)
rank_outer_frame.to_pickle(path=store_path + 'rank_outer')

num_params_frame = pd.DataFrame({'depth': depths, 'Num': jnp.array(num_params)}, dtype=float)
num_params_frame.to_pickle(path=store_path + 'num_params')

preds_F_frame = pd.DataFrame({'depth': depths, 'Pred': jnp.array(preds['rank_F'])}, dtype=float)
preds_F_frame.to_pickle(path=store_path + 'pred_F')
preds_L_frame = pd.DataFrame({'depth': depths, 'Pred': jnp.array(preds['rank_L'])}, dtype=float)
preds_L_frame.to_pickle(path=store_path + 'pred_L')
preds_outer_frame = pd.DataFrame({'depth': depths, 'Pred': jnp.array(preds['rank_outer'])}, dtype=float)
preds_outer_frame.to_pickle(path=store_path + 'pred_outer')
