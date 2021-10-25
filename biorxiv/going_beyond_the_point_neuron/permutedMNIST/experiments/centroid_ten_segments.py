#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#


"""
Experiment file that runs dendritic networks which infer the context vector via
prototyping, but use exactly ten dendritic segments per neuron regardless of the number
of tasks. This way, the number of model parameters stays constant as the number of
tasks grow.
"""


import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST

from .centroid import CentroidExperiment

BASE = dict(
    experiment_class=CentroidExperiment,
    num_samples=1,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        seed=42,
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=10,  # `num_segments` always stays fixed at 10 segments
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.05,
    ),

    batch_size=256,
    val_batch_size=512,
    tasks_to_validate=[1, 4, 9, 24, 49, 99],
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
)


# 2 permutedMNIST tasks, 10 dendritic segments per neuron
CENTROID_2_SEGMENTS_10 = deepcopy(BASE)
CENTROID_2_SEGMENTS_10["dataset_args"].update(num_tasks=2)
CENTROID_2_SEGMENTS_10.update(
    num_tasks=2,
    num_classes=10 * 2,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=1e-3)
)


# 5 permutedMNIST tasks, 10 dendritic segments per neuron
CENTROID_5_SEGMENTS_10 = deepcopy(BASE)
CENTROID_5_SEGMENTS_10["dataset_args"].update(num_tasks=5)
CENTROID_5_SEGMENTS_10.update(
    num_tasks=5,
    num_classes=10 * 5,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=7e-4)
)


# 10 permutedMNIST tasks, 10 dendritic segments per neuron
CENTROID_10_SEGMENTS_10 = deepcopy(BASE)
CENTROID_10_SEGMENTS_10["dataset_args"].update(num_tasks=10)
CENTROID_10_SEGMENTS_10.update(
    num_tasks=10,
    num_classes=10 * 10,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=5e-4)
)


# 25 permutedMNIST tasks, 10 dendritic segments per neuron
CENTROID_25_SEGMENTS_10 = deepcopy(BASE)
CENTROID_25_SEGMENTS_10["dataset_args"].update(num_tasks=25)
CENTROID_25_SEGMENTS_10.update(
    num_tasks=25,
    num_classes=10 * 25,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=3e-4)
)


# 50 permutedMNIST tasks, 10 dendritic segments per neuron
CENTROID_50_SEGMENTS_10 = deepcopy(BASE)
CENTROID_50_SEGMENTS_10["dataset_args"].update(num_tasks=50)
CENTROID_50_SEGMENTS_10.update(
    num_tasks=50,
    num_classes=10 * 50,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=7e-5)
)


# 100 permutedMNIST tasks, 10 dendritic segments per neuron
CENTROID_100_SEGMENTS_10 = deepcopy(BASE)
CENTROID_100_SEGMENTS_10["dataset_args"].update(num_tasks=100)
CENTROID_100_SEGMENTS_10.update(
    num_tasks=100,
    num_classes=10 * 100,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=7e-5)
)


CONFIGS = dict(
    centroid_2_segments_10=CENTROID_2_SEGMENTS_10,
    centroid_5_segments_10=CENTROID_5_SEGMENTS_10,
    centroid_10_segments_10=CENTROID_10_SEGMENTS_10,
    centroid_25_segments_10=CENTROID_25_SEGMENTS_10,
    centroid_50_segments_10=CENTROID_50_SEGMENTS_10,
    centroid_100_segments_10=CENTROID_100_SEGMENTS_10
)
