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
These are baseline experiments testing how well standard sparse and dense networks
perform on permuted MNIST in a continual learning setting without any dendrites.
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
from nupic.research.frameworks.dendrites.modules.dendritic_layers import (
    ZeroSegmentDendriticLayer,
)
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins


class NoDendriteExperiment(mixins.RezeroWeights,
                           mixins.PermutedMNISTTaskIndices,
                           mixins.UpdateBoostStrength,
                           DendriteContinualLearningExperiment):
    pass


# Continual learning with sparse networks.
# Run two MNIST tasks for debugging
SPARSE_CL_2 = dict(
    experiment_class=NoDendriteExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=2,
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        seed=42,
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),

    batch_size=128,
    val_batch_size=512,
    epochs=1,
    tasks_to_validate=[0, 1, 2, 3, 4, 9, 24, 49, 74, 99],
    num_classes=10 * 2,
    num_tasks=2,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=1,  # Increase to run multiple experiments in parallel

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
    optimizer_args=dict(lr=5e-4),

    # For wandb
    env_config=dict(
        wandb=dict(
            entity="nupic-research",
            project="dendrite_baselines",
            name="sparse_cl_2",
            group="sparse_cl_2",
            notes="""
            Sparse network with continual learning
            """
        )
    ),

)

SPARSE_CL_50 = deepcopy(SPARSE_CL_2)
SPARSE_CL_50["dataset_args"].update(num_tasks=50)
SPARSE_CL_50["env_config"]["wandb"].update(
    name="sparse_cl_50",
    group="sparse_cl_50_optimized   ",
)
SPARSE_CL_50.update(
    num_tasks=50,
    num_classes=10 * 50,
    num_samples=1,
)

SPARSE_CL_50_NO_NORM = deepcopy(SPARSE_CL_50)
SPARSE_CL_50_NO_NORM["dataset_args"].update(normalize=False)
SPARSE_CL_50_NO_NORM["env_config"]["wandb"].update(
    name="sparse_cl_50_no_norm",
    group="sparse_cl_50_no_norm",
)


SPARSE_CL_100 = deepcopy(SPARSE_CL_2)
SPARSE_CL_100["dataset_args"].update(num_tasks=100)
SPARSE_CL_100["env_config"]["wandb"].update(
    name="sparse_cl_100",
    group="sparse_cl_100",
)
SPARSE_CL_100.update(
    num_tasks=100,
    num_classes=10 * 100,
    num_samples=1,
)

SPARSE_CL_100_NO_NORM = deepcopy(SPARSE_CL_100)
SPARSE_CL_100_NO_NORM["dataset_args"].update(normalize=False)
SPARSE_CL_100_NO_NORM["env_config"]["wandb"].update(
    name="sparse_cl_100_no_norm",
    group="sparse_cl_100_no_norm",
)


# Used for hyperparameter optimization. No wandb for now until we figure out multiple
# runs.
SPARSE_CL_10_SEARCH = deepcopy(SPARSE_CL_2)
SPARSE_CL_10_SEARCH["dataset_args"].update(num_tasks=10)
SPARSE_CL_10_SEARCH.pop("env_config")
SPARSE_CL_10_SEARCH.update(
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=tune.sample_from(
            lambda spec: np.random.choice([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.sample_from(
            lambda spec: np.random.choice([0.4, 0.5, 0.75, 0.9])),
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),
    epochs=tune.sample_from(
        lambda spec: int(np.random.choice([1, 2, 3]))),
    num_tasks=10,
    num_classes=10 * 10,
    num_samples=10,
    optimizer_args=dict(
        lr=tune.sample_from(
            lambda spec: np.random.choice([0.001, 0.0001, 0.00001])),
    ),
)


# Continual learning with fully dense networks.
DENSE_CL_2 = deepcopy(SPARSE_CL_2)
DENSE_CL_2.update(
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=False,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.0,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),
)

DENSE_CL_50 = deepcopy(DENSE_CL_2)
DENSE_CL_50["dataset_args"].update(num_tasks=50)
DENSE_CL_50.update(
    num_tasks=50,
    num_classes=10 * 50,

    # For wandb
    env_config=dict(
        wandb=dict(
            entity="nupic-research",
            project="dendrite_baselines",
            name="dense_cl_50",
            group="dense_cl_50",
            notes="""
        Dense network with continual learning
        """
        )
    ),
)

DENSE_CL_50_NO_NORM = deepcopy(DENSE_CL_50)
DENSE_CL_50_NO_NORM["dataset_args"].update(normalize=False)
DENSE_CL_50_NO_NORM["env_config"]["wandb"].update(
    name="dense_cl_50_no_norm",
    group="dense_cl_50_no_norm",
)


DENSE_CL_100 = deepcopy(DENSE_CL_2)
DENSE_CL_100["dataset_args"].update(num_tasks=100)
DENSE_CL_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    # For wandb
    env_config=dict(
        wandb=dict(
            entity="nupic-research",
            project="dendrite_baselines",
            name="dense_cl_100",
            group="dense_cl_100",
            notes="""
        Dense network with continual learning
        """
        )
    ),
)


# Export all configurations in this file
CONFIGS = dict(
    sparse_cl_2=SPARSE_CL_2,
    sparse_cl_10_search=SPARSE_CL_10_SEARCH,
    sparse_cl_50=SPARSE_CL_50,
    sparse_cl_50_no_norm=SPARSE_CL_50_NO_NORM,
    sparse_cl_100=SPARSE_CL_100,
    sparse_cl_100_no_norm=SPARSE_CL_100_NO_NORM,

    dense_cl_2=DENSE_CL_2,
    dense_cl_50=DENSE_CL_50,
    dense_cl_50_no_norm=DENSE_CL_50_NO_NORM,
    dense_cl_100=DENSE_CL_100,
)
