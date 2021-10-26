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
Experiment file that runs Active Dendrites Networks which 1) construct a prototype
context vector during training, and 2) try to infer the correct prototype for each task
during inference.
"""

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.dendrites.mixins import CentroidFigure1B
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins
from nupic.torch.modules import KWinners


class CentroidExperiment(mixins.RezeroWeights,
                         mixins.CentroidContext,
                         mixins.PermutedMNISTTaskIndices,
                         DendriteContinualLearningExperiment):
    pass


class CentroidFigure1BExperiment(CentroidFigure1B,
                                 mixins.PlotHiddenActivations,
                                 CentroidExperiment):
    pass


CENTROID_BASE = dict(
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

    model_class=DendriticMLP,  # CentroidDendriticMLP does not affect accuracy..??
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    tasks_to_validate=[1, 4, 9, 24, 49, 99],
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
)


CENTROID_2 = deepcopy(CENTROID_BASE)
CENTROID_2["dataset_args"].update(num_tasks=2)
CENTROID_2["model_args"].update(num_segments=2)
CENTROID_2.update(
    num_tasks=2,
    num_classes=10 * 2,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=1,
    optimizer_args=dict(lr=5e-4),
)


CENTROID_5 = deepcopy(CENTROID_BASE)
CENTROID_5["dataset_args"].update(num_tasks=5)
CENTROID_5["model_args"].update(num_segments=5)
CENTROID_5.update(
    num_tasks=5,
    num_classes=10 * 5,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=1,
    optimizer_args=dict(lr=5e-4),
)


CENTROID_10 = deepcopy(CENTROID_BASE)
CENTROID_10["dataset_args"].update(num_tasks=10)
CENTROID_10["model_args"].update(num_segments=10)
CENTROID_10.update(
    num_tasks=10,
    num_classes=10 * 10,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=5e-4),
)


# This experiment configuration is for visualizing the hidden activations in an Active
# Dendrites Network on a per-task basis; it produces `.pt` files which can then be used
# by the hidden activations script to generate visualizations
HIDDEN_ACTIVATIONS_PER_TASK = deepcopy(CENTROID_10)
HIDDEN_ACTIVATIONS_PER_TASK.update(
    experiment_class=CentroidFigure1BExperiment,

    plot_hidden_activations_args=dict(
        include_modules=[KWinners],
        plot_freq=1,
        max_samples_to_plot=5000
    ),
)


CENTROID_25 = deepcopy(CENTROID_BASE)
CENTROID_25["dataset_args"].update(num_tasks=25)
CENTROID_25["model_args"].update(num_segments=25)
CENTROID_25.update(
    num_tasks=25,
    num_classes=10 * 25,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=3e-4),
)


CENTROID_50 = deepcopy(CENTROID_BASE)
CENTROID_50["dataset_args"].update(num_tasks=50)
CENTROID_50["model_args"].update(num_segments=50)
CENTROID_50.update(
    num_tasks=50,
    num_classes=10 * 50,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=3e-4),
)


CENTROID_100 = deepcopy(CENTROID_BASE)
CENTROID_100["dataset_args"].update(num_tasks=100)
CENTROID_100["model_args"].update(num_segments=100)
CENTROID_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=1e-4),
)


# Export configurations in this file
CONFIGS = dict(
    centroid_2=CENTROID_2,
    centroid_5=CENTROID_5,
    centroid_10=CENTROID_10,
    hidden_activations_per_task=HIDDEN_ACTIVATIONS_PER_TASK,
    centroid_25=CENTROID_25,
    centroid_50=CENTROID_50,
    centroid_100=CENTROID_100,
)
