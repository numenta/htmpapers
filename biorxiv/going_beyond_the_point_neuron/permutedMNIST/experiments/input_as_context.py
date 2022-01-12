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
Experiment file that runs dendritic networks which use the raw image as context
"""

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.continual_learning import mixins as cl_mixins
from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.dendrites.mixins import InputAsContext
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins as vernon_mixins


class SimpleExperiment(InputAsContext,
                       vernon_mixins.RezeroWeights,
                       cl_mixins.PermutedMNISTTaskIndices,
                       DendriteContinualLearningExperiment):
    pass


INPUT_AS_CONTEXT_10 = dict(
    experiment_class=SimpleExperiment,
    num_samples=1,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=10,
        root=os.path.expanduser("~/nta/results/data/"),
        download=True,  # Change to True if running for the first time
        seed=42,
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=10,
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=tune.grid_search([1, 2, 4, 8]),
    tasks_to_validate=[0, 1, 2, 3, 4, 9, 24, 49, 74, 99],
    num_tasks=10,
    num_classes=10 * 10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
    optimizer_args=dict(lr=tune.grid_search([1e-6, 1e-5, 1e-4, 5e-4])),
)


# Used to verify that input_as_context mixin produces the same results as
# previous implementation with a separate dataloader.
INPUT_AS_CONTEXT_10_REGRESSION_TEST = deepcopy(INPUT_AS_CONTEXT_10)
INPUT_AS_CONTEXT_10_REGRESSION_TEST.update(
    epochs=4,
    optimizer_args=dict(lr=.0001),
)


# 4 epochs and lr=.0001 were the best parameters
INPUT_AS_CONTEXT_10_ = deepcopy(INPUT_AS_CONTEXT_10)
INPUT_AS_CONTEXT_10_.update(
    num_samples=4,
    epochs=4,
    optimizer_args=dict(lr=.0001)
)


# Fix best hyperparameters and increase the number of tasks to 25, 50, 100
INPUT_AS_CONTEXT_25_ = deepcopy(INPUT_AS_CONTEXT_10_)
INPUT_AS_CONTEXT_25_.update(
    num_tasks=25
)
INPUT_AS_CONTEXT_25_["dataset_args"].update(
    num_tasks=25
)

INPUT_AS_CONTEXT_50_ = deepcopy(INPUT_AS_CONTEXT_10_)
INPUT_AS_CONTEXT_50_.update(
    num_tasks=50
)
INPUT_AS_CONTEXT_50_["dataset_args"].update(
    num_tasks=50
)

INPUT_AS_CONTEXT_100_ = deepcopy(INPUT_AS_CONTEXT_10_)
INPUT_AS_CONTEXT_100_.update(
    num_tasks=100
)
INPUT_AS_CONTEXT_100_["dataset_args"].update(
    num_tasks=100
)

# Export configurations in this file
CONFIGS = dict(
    input_as_context_10_regression_test=INPUT_AS_CONTEXT_10_REGRESSION_TEST,
    input_as_context_10=INPUT_AS_CONTEXT_10,
    input_as_context_10_=INPUT_AS_CONTEXT_10_,
    input_as_context_25_=INPUT_AS_CONTEXT_25_,
    input_as_context_50_=INPUT_AS_CONTEXT_50_,
    input_as_context_100_=INPUT_AS_CONTEXT_100_,
)
