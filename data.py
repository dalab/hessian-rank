from jax import random
import jax
import jax.numpy as np
import numpy as onp
import torchvision
import torch
import os


class MixtureGaussian:
    def __init__(self, n_train, n_test, dim, classes, key):
        """
        Mixture of Gaussian dataset, create classes by assigning each class a different mean and then sample from
        Gaussian to create inputs, one-hot encode targets.

        :param n_train:     int, number of training examples
        :param n_test:      int, number of test examples
        :param dim:         int, dimensionality of inputs
        :param classes:     int, number of classes, i.e. cluster centers
        :param key:         key, used to generate randomness
        """
        self.n_train = n_train
        self.n_test = n_test
        self.dim = dim
        self.classes = classes
        self.key = key

        self.x_train, self.y_train, self.x_test, self.y_test = self.get_data()

    def get_data(self):
        n_per_class_train = (self.n_train + 1) // self.classes
        n_per_class_test = (self.n_test + 1) // self.classes

        # Sample random centers
        key, self.key = random.split(self.key, 2)
        c = random.normal(key=key, shape=(self.classes, self.dim))

        key, self.key = random.split(self.key, 2)
        x_trains = []
        y_trains = []
        x_tests = []
        y_tests = []

        for i in range(self.classes):
            # Create training input and targets with different means
            x_train = random.normal(key, shape=(n_per_class_train, self.dim)) + np.reshape(c[i, :], (1, -1))
            x_trains.append(x_train)
            y_trains += [i for _ in range(n_per_class_train)]

            # Create test input and targets with different means
            x_test = random.normal(key, shape=(n_per_class_test, self.dim)) + np.reshape(c[i, :], (1, -1))
            x_tests.append(x_test)
            y_tests += [i for _ in range(n_per_class_test)]

        x_train = np.concatenate(x_trains, axis=0)
        y_train = jax.nn.one_hot(y_trains, self.classes)
        x_test = np.concatenate(x_tests, axis=0)
        y_test = jax.nn.one_hot(y_tests, self.classes)
        x_train = jax.numpy.float64(x_train)
        x_test = jax.numpy.float64(x_test)

        return x_train, y_train, x_test, y_test

    def get_emp_cov(self):
        """Calculates empirical covariance and cross-covariance"""
        emp_cov = 1 / self.n_train * self.x_train.T @ self.x_train
        cross_cov = 1 / self.n_train * self.y_train.T @ self.x_train

        return emp_cov, cross_cov


class TinyMNIST:
    def __init__(self, n_train, n_test, d, classes=10, flat=True, key=None, noise_level=None):
        """
        Implements data structure for MNIST, allowing to resize the images

        :param n_train:         int, number of training examples
        :param n_test:          int, number of test examples
        :param dim:             int, dimensionality of inputs, images are rescaled according to d
        :param classes:         int, number of classes, one of '2' or '10'
        :param flat:            bool, flatten image to vector
        :param key:             key, used to generate randomness
        :param noise_level:     float, between 0 and 1, specifying noise level in targets
        """
        self.n_train = n_train
        self.n_test = n_test
        self.d = d
        self.flat = flat
        self.key = key
        self.noise_level = noise_level
        self.classes = classes

        self.get_data()

        if noise_level is not None:
            random_key, self.key = random.split(self.key)
            self.randomize(random_key)

    def get_data(self):
        # Load and store data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        trainset = torchvision.datasets.MNIST(root=dir_path + '/data', train=True, download=True)
        testset = torchvision.datasets.MNIST(root=dir_path + '/data', train=False, download=True)

        # Resize the data through interpolation
        self.x_train = torch.unsqueeze(trainset.train_data[:, :, :], 1) / 255.0
        self.x_train = torch.nn.functional.interpolate(self.x_train, (self.d, self.d)).squeeze().numpy()
        self.x_test = torch.unsqueeze(testset.train_data[:, :, :], 1) / 255.0
        self.x_test = torch.nn.functional.interpolate(self.x_test, (self.d, self.d)).squeeze().numpy()

        self.y_train = onp.expand_dims(trainset.train_labels.numpy(), axis=1)
        self.y_test = onp.expand_dims(testset.train_labels.numpy(), axis=1)

        if self.classes == 2:
            where = self.y_train < 2
            indices = np.where(where > 0)
            self.x_train = self.x_train[indices[0], :, :]
            self.y_train = 2 * (self.y_train[indices[0]] - 1/2)

        self.x_train = self.x_train[:self.n_train, :, :]
        self.y_train = self.y_train[:self.n_train]
        self.x_test = self.x_test[:self.n_test, :, :]
        self.y_test = self.y_test[:self.n_test]

        if self.classes != 2:
            self.y_train = onp.expand_dims(self.y_train[:self.n_train], axis=1)

        if self.classes != 2:
            # If we use more than two classes, one-hot encode instead of -1, 1 targets
            self.y_train = jax.nn.one_hot(self.y_train, self.classes).squeeze()
            self.y_test = jax.nn.one_hot(self.y_test, self.classes).squeeze()

        if self.flat:
            # Flatten the data to vectors to use fully-connected architectures
            self.x_train = onp.reshape(self.x_train, (self.n_train, -1))
            self.x_test = onp.reshape(self.x_test, (self.n_test, -1))

        # Use float64 to guarantee accurate rank calculations
        self.x_train = jax.numpy.float64(self.x_train)
        self.x_test = jax.numpy.float64(self.x_test)

    def add_data(self, num_samples):
        """Add num_samples to dataset"""
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False)

        x_train_new = torch.unsqueeze((trainset.train_data[self.n_train:(self.n_train + num_samples), :, :] / 255.0),
                                       dim=1)
        x_train_new = torch.nn.functional.interpolate(x_train_new, (self.d, self.d)).squeeze().numpy()

        y_train_new = onp.expand_dims(trainset.train_labels[self.n_train:(self.n_train + num_samples)].numpy(),
                                       axis=1)
        y_train_new = jax.nn.one_hot(y_train_new, self.classes).squeeze()

        self.n_train += num_samples
        if self.flat:
            x_train_new = onp.reshape(x_train_new, (num_samples, -1))
        x_train_new = jax.numpy.float64(x_train_new)
        self.x_train = np.concatenate([self.x_train, x_train_new], axis=0)
        self.y_train = np.concatenate([self.y_train, y_train_new], axis=0)

    def sub_classes(self):
        """Create subclasses in case of K < 10"""
        use_classes = np.array([i for i in range(self.classes)])
        where = self.y_train == use_classes
        where = np.sum(where, axis=1)
        indices = np.where(where > 0)

        self.x_train = self.x_train[indices[0], :]
        self.y_train = self.y_train[indices[0], ]
        self.y_train = jax.nn.one_hot(self.y_train, self.classes).squeeze()

        self.n_train = self.x_train.shape[0]

    def randomize(self, key):
        """Randomize a portion of the labels, according to 'noise_level'"""
        key, new_key = random.split(key, 2)
        indices = random.bernoulli(key, shape=(self.n_train, 1), p=self.noise_level)
        noise = random.bernoulli(key, shape=(self.n_train, 1), p=0.5)
        y_train_noisy = self.y_train - 2 * noise * indices * self.y_train
        self.correct_indices, _ = np.where(y_train_noisy == self.y_train)
        self.wrong_indices, _ = np.where(y_train_noisy != self.y_train)
        self.y_train = y_train_noisy

    def get_emp_cov(self):
        """Calculate empirical covariance and cross covariance"""
        emp_cov = self.x_train.T @ self.x_train
        cross_cov = self.y_train.T @ self.x_train

        return emp_cov, cross_cov


class TinyFashionMNIST:
    def __init__(self, n_train, n_test, d, classes=10, flat=True, key=None, noise_level=None):
        """
        Implements data structure for FashionMNIST, allowing to resize the images

        :param n_train:         int, number of training examples
        :param n_test:          int, number of test examples
        :param dim:             int, dimensionality of inputs, images are rescaled according to d
        :param classes:         int, number of classes, one of '2' or '10'
        :param flat:            bool, flatten image to vector
        :param key:             key, used to generate randomness
        :param noise_level:     float, between 0 and 1, specifying noise level in targets
        """
        self.n_train = n_train
        self.n_test = n_test
        self.d = d
        self.flat = flat
        self.key = key
        self.noise_level = noise_level
        self.classes = classes

        self.get_data()

        if self.classes < 10 and self.classes > 1:
            self.sub_classes()

        if noise_level is not None:
            random_key, self.key = random.split(self.key)
            self.randomize(random_key)

    def get_data(self):
        # Load and store data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)
        trainset = torchvision.datasets.FashionMNIST(root=dir_path + '/data', train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(root=dir_path + '/data', train=False, download=True)

        # Resize the data through interpolation
        self.x_train = torch.unsqueeze(trainset.train_data[:, :, :], 1) / 255.0
        self.x_train = torch.nn.functional.interpolate(self.x_train, (self.d, self.d)).squeeze().numpy()
        self.y_train = onp.expand_dims(trainset.train_labels.numpy(), axis=1)

        if self.classes == 2:
            where = self.y_train < 2
            indices = np.where(where > 0)
            self.x_train = self.x_train[indices[0], :, :]
            self.y_train = 2 * (self.y_train[indices[0]] - 1 / 2)

        self.x_train = self.x_train[:self.n_train, :, :]
        self.y_train = self.y_train[:self.n_train]
        if self.classes != 2:
            self.y_train = onp.expand_dims(self.y_train[:self.n_train], axis=1)
        self.x_test = onp.expand_dims(testset.test_data[:self.n_test, :, :].numpy(), 3) / 255.0

        if self.classes != 2:
            # If we use more than two classes, one-hot encode instead of -1, 1 targets
            self.y_train = jax.nn.one_hot(self.y_train, self.classes).squeeze()
            self.y_test = onp.expand_dims(testset.test_labels[:self.n_test].numpy(), axis=1)
            self.y_test = jax.nn.one_hot(self.y_test, self.classes).squeeze()

        if self.flat:
            # Flatten the data to vectors to use fully-connected architectures
            self.x_train = onp.reshape(self.x_train, (self.n_train, -1))
            self.x_test = onp.reshape(self.x_test, (self.n_test, -1))

        # Use float64 to guarantee accurate rank calculations
        self.x_train = jax.numpy.float64(self.x_train)
        self.x_test = jax.numpy.float64(self.x_test)

    def add_data(self, num_samples):
        """Add num_samples to dataset"""
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)

        x_train_new = torch.unsqueeze((trainset.train_data[self.n_train:(self.n_train + num_samples), :, :] / 255.0),
                                       dim=1)
        x_train_new = torch.nn.functional.interpolate(x_train_new, (self.d, self.d)).squeeze().numpy()

        y_train_new = onp.expand_dims(trainset.train_labels[self.n_train:(self.n_train + num_samples)].numpy(),
                                       axis=1)
        y_train_new = jax.nn.one_hot(y_train_new, self.classes).squeeze()

        self.n_train += num_samples
        if self.flat:
            x_train_new = onp.reshape(x_train_new, (num_samples, -1))
        x_train_new = jax.numpy.float64(x_train_new)
        self.x_train = np.concatenate([self.x_train, x_train_new], axis=0)
        self.y_train = np.concatenate([self.y_train, y_train_new], axis=0)

    def sub_classes(self):
        """Create subclasses in case of K < 10"""
        use_classes = np.array([i for i in range(self.classes)])
        where = self.y_train == use_classes
        where = np.sum(where, axis=1)
        indices = np.where(where > 0)

        self.x_train = self.x_train[indices[0], :]
        self.y_train = self.y_train[indices[0], ]
        self.y_train = jax.nn.one_hot(self.y_train, self.classes).squeeze()

        self.n_train = self.x_train.shape[0]

    def randomize(self, key):
        """Randomize a portion of the labels, according to 'noise_level'"""
        key, new_key = random.split(key, 2)
        indices = random.bernoulli(key, shape=(self.n_train, 1), p=self.noise_level)
        noise = random.bernoulli(key, shape=(self.n_train, 1), p=0.5)
        y_train_noisy = self.y_train - 2 * noise * indices * self.y_train
        self.correct_indices, _ = np.where(y_train_noisy == self.y_train)
        self.wrong_indices, _ = np.where(y_train_noisy != self.y_train)
        self.y_train = y_train_noisy

    def get_emp_cov(self):
        emp_cov = self.x_train.T @ self.x_train
        cross_cov = self.y_train.T @ self.x_train

        return emp_cov, cross_cov


class TinyCIFAR10:
    def __init__(self, n_train, n_test, d, classes=10, flat=True, key=None):
        """
        Implements data structure for CIFAR10, allowing to resize the images

        :param n_train:         int, number of training examples
        :param n_test:          int, number of test examples
        :param dim:             int, dimensionality of inputs, images are rescaled according to d
        :param classes:         int, number of classes, one of '2' or '10'
        :param flat:            bool, flatten image to vector
        :param key:             key, used to generate randomness
        """
        self.n_train = n_train
        self.n_test = n_test
        self.flat = flat
        self.d = d
        self.key = key
        self.classes = classes

        self.get_data()

    def get_data(self):
        # Load and store data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        trainset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=False, download=True)

        # Resize the data through interpolation
        self.x_train = torch.tensor(trainset.data[:, :, :]) / 255.0
        self.x_train = torch.transpose(self.x_train, 1, 3)
        self.x_train = torch.nn.functional.interpolate(self.x_train, (self.d, self.d)).squeeze().numpy()
        self.y_train = onp.expand_dims(trainset.targets, axis=1)

        if self.classes == 2:
            where = self.y_train < 2
            indices = np.where(where > 0)
            self.x_train = self.x_train[indices[0], :, :]
            self.y_train = 2 * (self.y_train[indices[0]] - 1 / 2)

        self.x_train = self.x_train[:self.n_train, :, :]
        self.y_train = self.y_train[:self.n_train]

        if self.classes != 2:
            self.y_train = onp.expand_dims(self.y_train[:self.n_train], axis=1)
        self.x_test = onp.expand_dims(testset.data[:self.n_test, :, :], 3) / 255.0

        if self.classes != 2:
            # If we use more than two classes, one-hot encode instead of -1, 1 targets
            self.y_train = jax.nn.one_hot(self.y_train, self.classes).squeeze()

        if self.flat:
            # Flatten the data to vectors to use fully-connected architectures
            self.x_train = self.x_train[:, 0, :, :]
            self.x_train = onp.reshape(self.x_train, (self.n_train, -1))
            self.x_test = self.x_test[:, 0, :, :]
            self.x_test = onp.reshape(self.x_test, (self.n_test, -1))

        # Use float64 to guarantee accurate rank calculations
        self.x_train = jax.numpy.float64(self.x_train)
        self.x_test = jax.numpy.float64(self.x_test)

    def get_emp_cov(self):
        """Calculate empirical covariance and cross covariance"""
        emp_cov = self.x_train.T @ self.x_train
        cross_cov = self.y_train.T @ self.x_train

        return emp_cov, cross_cov


def get_dataset(name, n_train, n_test, dim, classes):
    """
    Helper function to load the desired dataset.
    :param dataset:     str, one 'MNIST', 'CIFAR', 'FashionMNIST'
    :param n_train:     int, number of training examples
    :param n_test:      int, number of test examples
    :param dim:         int, desired input dimensionality achieved through rescaling
    :param classes:     int, desired number of classes, one of '2' or '10'

    :return:            data, object of class dataset
    """
    if name == 'MNIST':
        return TinyMNIST(n_train=n_train, n_test=n_test, d=int(np.sqrt(dim)), classes=classes)
    if name == 'CIFAR':
        return TinyCIFAR10(n_train=n_train, n_test=n_test, d=int(np.sqrt(dim)), classes=classes)
    if name == 'FashionMNIST':
        return TinyFashionMNIST(n_train=n_train, n_test=n_test, d=int(np.sqrt(dim)), classes=classes)