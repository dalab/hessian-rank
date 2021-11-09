import torch
from torch.utils.data import Dataset
import numpy as np
import jax.lax as lax


class DatasetTorch(Dataset):
    def __init__(self, inputs, labels):
        """
        Data structure to enable efficient batching based on torch's Dataset structure
        :param inputs:      jnp.array, storing the inputs
        :param labels:      jnp.array, storing the targets
        """
        self.inputs = torch.from_numpy(np.array(inputs))
        self.labels = torch.from_numpy(np.array(labels))

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input_batch = self.inputs[idx, :]
        label_batch = self.labels[idx, :]

        return input_batch, label_batch
