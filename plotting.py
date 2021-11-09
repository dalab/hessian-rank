import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle5 as pickle
from plot_utils import HandlerRect, HandlerCircle
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--loss', default='mse', type=str)
parser.add_argument('--dataset', default='MNIST', type=str)
parser.add_argument('--task', default='depth', type=str)
parser.add_argument('--init', default='glorot', type=str)
parser.add_argument('--normalize', default=1, type=int)
parser.add_argument('--width', default=None, type=int)
parser.add_argument('--units', default=None, type=str)
parser.add_argument('--dim', default=None, type=int)
parser.add_argument('--K', default=None, type=int)

args = parser.parse_args()

# If normalize=0, just plot the ranks, if normalize=1 plot rank/#params
normalize = args.normalize == 1

# Load all the data
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/results/store/' + args.task + '/' + args.loss + '/'\
           + args.dataset + '/'

if args.task == 'depth':
    dir_path += str(args.width) + '/'

if args.task == 'samplesize':
    m = args.units.split(',')
    depth = len(m)
    unit_string = str(args.dim) + 'x'
    for i in range(depth):
        unit_string += str(m[i]) + 'x'
    unit_string += str(args.K)
    dir_path += unit_string + '/'


task = args.task

with open(dir_path + 'rank_F', "rb") as fh:
    rank_F_frame = pickle.load(fh)

with open(dir_path + 'rank_L', "rb") as fh:
    rank_L_frame = pickle.load(fh)

with open(dir_path + 'rank_outer', "rb") as fh:
    rank_outer_frame = pickle.load(fh)

with open(dir_path + 'pred_F', "rb") as fh:
    preds_F_frame = pickle.load(fh)

with open(dir_path + 'pred_L', "rb") as fh:
    preds_L_frame = pickle.load(fh)

with open(dir_path + 'pred_outer', "rb") as fh:
    preds_outer_frame = pickle.load(fh)

if task == 'width' or task == 'depth':
    with open(dir_path + 'num_params', "rb") as fh:
        if normalize:
            num_params = pickle.load(fh)
            rank_F_frame['Rank'] = rank_F_frame['Rank'] / num_params['Num']
            rank_L_frame['Rank'] = rank_L_frame['Rank'] / num_params['Num']
            rank_outer_frame['Rank'] = rank_outer_frame['Rank'] / num_params['Num']
            preds_F_frame['Pred'] = preds_F_frame['Pred'] / num_params['Num']
            preds_L_frame['Pred'] = preds_L_frame['Pred'] / num_params['Num']
            preds_outer_frame['Pred'] = preds_outer_frame['Pred'] / num_params['Num']


# The plotting starts here
sns.set(font_scale=1, rc={'text.usetex': True, 'text.latex.preamble': [r"""\usepackage{bm}""",
                                                                       r"""\usepackage{amsmath}"""]})
sns.set_style('whitegrid')
if task == 'samplesize':
    name = 'n'
else:
    name = task

# Plot the empirical ranks as lines
sns.lineplot(data=rank_outer_frame, x=name, y='Rank', color='#ffa600', zorder=1, linewidth=1.5)
sns.lineplot(data=rank_F_frame, x=name, y='Rank', color='#ff6361', zorder=1, linewidth=1.5, alpha=1)
sns.lineplot(data=rank_L_frame, x=name, y='Rank', color='#bc5090', zorder=1, linewidth=1.5)

# Plot the prediction form the formulas as dots
sns.scatterplot(data=preds_outer_frame, x=name, y='Pred', color='#ff5e00', zorder=50, s=27, alpha=1)
sns.scatterplot(data=preds_F_frame, x=name, y='Pred', color='#f23d3d', zorder=50, s=27, alpha=1)
sns.scatterplot(data=preds_L_frame, x=name, y='Pred', color='#8c3c6b', zorder=50, s=27)

# Use rectangles for empirical rank and circles for the predictions
rect3 = patches.Rectangle((0, 0), 1, 1, facecolor='#ffa600')
rect2 = patches.Rectangle((0, 0), 1, 1, facecolor='#ff6361')
rect1 = patches.Rectangle((0, 0), 1, 1, facecolor='#bc5090')
circ3 = patches.Circle((0, 0), radius=1, facecolor='#ff5e00')
circ2 = patches.Circle((0, 0), radius=1, facecolor='#f23d3d')
circ1 = patches.Circle((0, 0), radius=1, facecolor='#8c3c6b')

# Create x and y axes
if task == 'width':
    plt.xlabel('Minimal Width')
    if normalize:
        plt.ylabel('Rank / ' + r'$\# \text{params}$')
    else:
        plt.ylabel('Rank')

elif task == 'depth':
    plt.xlabel('Depth')
    plt.ylabel('Rank / ' + r'$\# \text{params}$')

elif task == 'samples':
    plt.xlabel('Number of Samples')
    plt.ylabel('Rank')

# Add legends
plt.legend(labels=[r'$rank(\bm{H}_{\mathcal{L}})$',  r'$Pred_{\mathcal{L}}$', r'$rank(\bm{H}_f)$', r'$Pred_F$',
                   r'$rank(\bm{H}_{o})$',
                   r'$Pred_{o}$'],
           loc='upper left',
           bbox_to_anchor=(0.0, 1.165),
           ncol=3, fancybox=False, shadow=False, handles=(rect1, circ1, rect2, circ2, rect3, circ3), handler_map={
               patches.Rectangle: HandlerRect(), patches.Circle: HandlerCircle()}, frameon=False)

# Save the figure
plt.savefig(dir_path + task + str(normalize), dpi=500)
plt.show()