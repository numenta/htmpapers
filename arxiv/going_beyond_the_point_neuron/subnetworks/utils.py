#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# flake8: noqa
"""
Support functions for experiment.ipynb
"""


import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from nupic.research.frameworks.dendrites import DendriteSegments

__all__ = [
    "split_data",
    "apply_target_transform",
    "add_task_embedding",
    "prep_dataset",
    "load_datasets",
    "evaluate",
    "train",
    "run_experiment",
    "plot_activations",
    "StandardMLP",
    "DendriticMLP",
]


# ------ Dataset
def split_data(X, Y, random_state=None):
    """Split with random selection
    Return two tuples, one with X and another with Ys"""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state, stratify=Y)
    return (X_train, X_test), (Y_train, Y_test)


def apply_target_transform(Y, offset=0):
    """Transform label for continual learning problems"""
    return [label + offset for label in Y]


def add_task_embedding(X, task_num=1, units=2):
    """Adds a task embedding to the input
    One hot encodes a task and append it to the input variables
    of every sample.

    In one hot encoding, all variables are set to 0, except the one
    corresponding to the task number, which is set to 1.

    :param units:
        Number of bits in the encoding vector
    :param task_num:
        Determines which bit of the encoding is set to 1

    Example:
    If there are 6 units in the vector, and task_num is 1, the
    vector [0, 1, 0, 0, 0, 0] is appended to the input vector
    of every sample
    """
    assert task_num < units, \
        "Task num are 0-indexed should be equal to max units-1"

    num_rows = X.shape[0]
    embedding = np.zeros((num_rows, units))
    embedding[:, task_num] = 1
    return np.hstack([X, embedding])


def prep_dataset(datasets, embed_task=False, random_state=None,
                 target_transform=False, target_transform_offset=3):
    """Accepts an array of tuples (X, Y, name) or a single tuple
    Return an array of tuples (Xs, Ys, name) or a single tuple
    where Xs and Ys are also a tuple containing data for training and test"""

    if type(datasets) != list:
        datasets = [datasets]

    ready_datasets = []
    for idx, (X, Y, dataset_name) in enumerate(datasets):
        if target_transform:
            Y = apply_target_transform(Y, target_transform_offset * idx)
        if embed_task:
            X = add_task_embedding(X, task_num=idx)
        Xs, Ys = split_data(X, Y, random_state=random_state)
        ready_datasets.append((Xs, Ys, dataset_name))

    if len(ready_datasets) == 1:
        return ready_datasets[0]
    return ready_datasets


def load_datasets():
    """Load Iris, Wine and Mixed datasets
    Returns three tuples, one for each dataset.
    Each tuple contains (data, target, dataset_name)
    data and target are numpy arrays, and dataset_name a string.
    """
    # load datsets
    iris = datasets.load_iris()
    wine = datasets.load_wine()

    # rebalance Wine dataset to have only 150 sample, same as iris
    wine_indices_class0 = np.random.choice(range(59), size=51, replace=False)
    wine_indices_class1 = np.random.choice(range(59, 59 + 71), size=51, replace=False)
    wine_indices_class2 = np.array(range(59 + 71, 178))
    wine_indices = np.sort(np.concatenate([
        wine_indices_class0, wine_indices_class1, wine_indices_class2
    ]))

    # select 4 most relevant features, using feature selection methods:
    # proline 12, od280/od315_of_diluted_wines 11
    # flavanoids 6, color_intensity 9
    wineX = wine.data[:, [6, 9, 11, 12]][wine_indices]

    # scale both datasets from 0 to 1
    irisX = MinMaxScaler().fit_transform(iris.data)
    irisY = iris.target
    wineX = MinMaxScaler().fit_transform(wineX)
    wineY = wine.target[wine_indices]

    # create mixed dataset
    mixedX = np.concatenate([irisX, wineX])
    mixedY = np.concatenate([irisY, apply_target_transform(wineY, 3)])

    return (irisX, irisY, "Iris"), (wineX, wineY, "Wine"), (mixedX, mixedY, "Mixed")


# ------ Training functions
def evaluate(model, data, target, dataset_name=""):
    pred = torch.argmax(model(torch.FloatTensor(data)), dim=1).numpy()
    print(f"{dataset_name} Acc: {accuracy_score(target, pred):.4f}")


def train(model, data, target, dataset_name, batch_size=1,
          epochs=10, lr=1e-2, weight_decay=0, verbose=True):

    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    if verbose:
        print("\n")
        print(f"Training {dataset_name} ...")

    # convert to tensors
    if not isinstance(data, torch.Tensor) and not isinstance(target, torch.Tensor):
        data = torch.FloatTensor(data)
        target = torch.LongTensor(target)

    for epoch in range(epochs):
        # randomize input
        indices = np.random.permutation(data.shape[0])

        # iterate through batchs
        for batch_idx in range(math.ceil(data.shape[0] / batch_size)):

            optim.zero_grad()

            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_data = data[batch_indices]
            batch_target = target[batch_indices]
            batch_pred = model(batch_data)

            loss = loss_fn(batch_pred, batch_target)
            loss.backward()
            optim.step()

        if (epoch + 1) % (epochs / 5) == 0 and verbose:
            pred = torch.argmax(model(torch.FloatTensor(data)), dim=1).numpy()
            print(f"Epoch: {epoch+1}  Acc: {accuracy_score(target.numpy(), pred):.4f}")


def run_experiment(model, datasets, num_loops=1, evaluate_training_set=True,
                   batch_size=10, epochs=60, lr=1e-2, weight_decay=0,
                   verbose=False):
    """Learn one task, then another, and evaluate forgetting on previous one
    Notes:
    - Order of the datasets matter
    - To run a regular experiments, just pass a single dataset
    """

    # account for case where single dataet is passed
    if type(datasets) != list:
        datasets = [datasets]

    # train from first to last
    for loop_idx in range(num_loops):
        if verbose and num_loops > 1:
            print(f"Loop {loop_idx}")
        for dataset in datasets:
            Xs, Ys, dataset_name = dataset
            train(model, Xs[0], Ys[0], dataset_name, batch_size=batch_size,
                  epochs=epochs, lr=lr, weight_decay=weight_decay,
                  verbose=verbose)

    # evaluate from last to first
    for dataset in datasets[::-1]:
        print("\n")
        Xs, Ys, dataset_name = dataset
        if evaluate_training_set:
            evaluate(model, Xs[0], Ys[0], dataset_name + " training set")
        evaluate(model, Xs[1], Ys[1], dataset_name + " test set")


# ------ Networks
class StandardMLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=(100, 100),
                 activation_fn=nn.ReLU):

        super().__init__()

        layers = [
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0], affine=False),
            activation_fn()
        ]

        for idx in range(1, len(hidden_sizes)):
            layers.extend([
                nn.Linear(hidden_sizes[idx - 1], hidden_sizes[idx]),
                nn.BatchNorm1d(hidden_sizes[idx], affine=False),
                activation_fn()
            ])
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.classifier = nn.Sequential(*layers)
        # self.init_weights()

    def forward(self, x):
        return self.classifier(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class DendriticMLP(nn.Module):
    """
    Based on MimicANMLDendriticNetwork, using dendritic output as [0,1] gates
    Added some modifications to inspect output and make it easier to debug
    and manipulate.
    """

    def __init__(self, input_size, output_size,
                 hidden_sizes=(10, 10),
                 dim_context=2,
                 num_segments=(5, 5, 5),
                 module_sparsity=(.75, .75, .75),
                 dendrite_sparsity=(.5, .5, .5),
                 dendrite_bias=(False, False, False),
                 activation_fn=nn.ReLU):
        super().__init__()

        self.input_size = input_size

        self.block0 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0], affine=False),
            activation_fn()
        )

        self.segments0 = DendriteSegments(
            num_units=hidden_sizes[0],
            num_segments=num_segments[0],
            dim_context=dim_context,
            sparsity=dendrite_sparsity[0],
            bias=dendrite_bias[0],
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1], affine=False),
            activation_fn()
        )

        self.segments1 = DendriteSegments(
            num_units=hidden_sizes[1],
            num_segments=num_segments[1],
            dim_context=dim_context,
            sparsity=dendrite_sparsity[1],
            bias=dendrite_bias[1],
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2], affine=False),
            activation_fn()
        )

        self.segments2 = DendriteSegments(
            num_units=hidden_sizes[2],
            num_segments=num_segments[2],
            dim_context=dim_context,
            sparsity=dendrite_sparsity[2],
            bias=dendrite_bias[2],
        )

        self.classifier = nn.Linear(hidden_sizes[-1], output_size)

    def apply_dendrites(self, y, dendrite_activations):
        """
        Apply dendrites as a gating mechanism.
        """
        # # Multiple by the sigmoid of the max along each segment.
        # return y * torch.sigmoid(dendrite_activations.max(dim=2).values)

        inds = dendrite_activations.abs().max(dim=2).indices
        inds = inds.unsqueeze(dim=2)
        dendrite_activations = torch.gather(dendrite_activations, dim=2, index=inds)
        dendrite_activations = dendrite_activations.squeeze(dim=2)
        dendrite_activations = torch.sigmoid(dendrite_activations)
        return y * dendrite_activations

    def forward(self, x):

        x_input = x[:, :self.input_size]
        context = x[:, self.input_size:]

        x = self.block0(x_input)
        x = self.apply_dendrites(x, self.segments0(context))

        x = self.block1(x)
        x = self.apply_dendrites(x, self.segments1(context))

        x = self.block2(x)
        x = self.apply_dendrites(x, self.segments2(context))

        return self.classifier(x)

    def reduce_dendrites(self, dendrite_activations):
        """
        Calculate gate for each unit as float from 0 to 1
        """
        # # Multiple by the sigmoid of the max along each segment.
        # return y * torch.sigmoid(dendrite_activations.max(dim=2).values)

        inds = dendrite_activations.abs().max(dim=2).indices
        inds = inds.unsqueeze(dim=2)
        dendrite_activations = torch.gather(dendrite_activations, dim=2, index=inds)
        dendrite_activations = dendrite_activations.squeeze(dim=2)
        dendrite_activations = torch.sigmoid(dendrite_activations)
        return dendrite_activations

    def get_act_maps(self, x):
        """Manual hooks to collect data on dendrites as well"""

        act_maps = {}

        x_input = x[:, :self.input_size]
        context = x[:, self.input_size:]
        n = x.shape[0]

        x = self.block0(x_input)
        dendrites0 = self.reduce_dendrites(self.segments0(context))
        act_maps["block0"] = (torch.sum(x, dim=0) / n).detach().numpy()
        act_maps["dendrites0"] = (torch.sum(dendrites0, dim=0) / n).detach().numpy()

        x = x * dendrites0

        x = self.block1(x)
        dendrites1 = self.reduce_dendrites(self.segments1(context))
        act_maps["block1"] = (torch.sum(x, dim=0) / n).detach().numpy()
        act_maps["dendrites1"] = (torch.sum(dendrites1, dim=0) / n).detach().numpy()

        x = x * dendrites1

        x = self.block2(x)
        dendrites2 = self.reduce_dendrites(self.segments2(context))
        act_maps["block2"] = (torch.sum(x, dim=0) / n).detach().numpy()
        act_maps["dendrites2"] = (torch.sum(dendrites2, dim=0) / n).detach().numpy()

        x = x * dendrites2

        return act_maps

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def plot_activations(model, datasets):

    if type(datasets) != list:
        datasets = [datasets]

    for dataset in datasets:
        Xs, Ys, dataset_name = dataset
        actmap = model.get_act_maps(torch.FloatTensor(np.concatenate(Xs)))

        # reorganize data - network is from bottom to top
        subnet = np.vstack([v for k, v in sorted(actmap.items(), reverse=True)
                            if "dendrites" in k])

        # plotting
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(subnet, cmap=plt.cm.Blues)
        for i in range(subnet.shape[0]):
            for j in range(subnet.shape[1]):
                ax.text(j, i, f"{subnet[i,j]:.2f}", va="center", ha="center")

        plt.title(dataset_name + "\n")
