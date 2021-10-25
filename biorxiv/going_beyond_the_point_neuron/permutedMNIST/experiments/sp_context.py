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
from nupic.research.frameworks.dendrites.mixins import SpatialPoolerContext
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins


class SPExperiment(mixins.RezeroWeights,
                   mixins.PermutedMNISTTaskIndices,
                   mixins.UpdateBoostStrength,
                   SpatialPoolerContext,
                   DendriteContinualLearningExperiment):
    pass


# Spatial pooler for inferring contexts: 10 permutedMNIST tasks
SP_CONTEXT_10 = dict(
    experiment_class=SPExperiment,
    num_samples=2,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=10,
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        seed=42,
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=14,
        dim_context=500,
        kw=True,
        kw_percent_on=0.1,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.05,
    ),

    context_model_args=dict(
        kw_percent_on=0.05,
        boost_strength=0.0,
        weight_sparsity=0.75,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=2,
    tasks_to_validate=(0, 1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 74, 99),
    num_tasks=10,
    num_classes=10 * 10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
    optimizer_args=dict(lr=5e-4),
)

# Two tasks only, for debugging
SP_CONTEXT_2 = deepcopy(SP_CONTEXT_10)
SP_CONTEXT_2["dataset_args"].update(num_tasks=2)
SP_CONTEXT_2["model_args"].update(num_segments=2)
SP_CONTEXT_2.update(
    epochs=2,
    num_samples=2,
    num_tasks=2,
    num_classes=10 * 2,
)

SP_CONTEXT_50 = deepcopy(SP_CONTEXT_10)
SP_CONTEXT_50["dataset_args"].update(num_tasks=50)
SP_CONTEXT_50["model_args"].update(
    num_segments=50,
)
SP_CONTEXT_50.update(
    epochs=2,
    num_samples=1,
    num_tasks=50,
    num_classes=10 * 50,
    batch_size=64,

    context_model_args=dict(
        kw_percent_on=0.05,
        boost_strength=1.0,
        weight_sparsity=0.5,
    ),

    optimizer_args=dict(lr=1e-4),

    env_config=dict(
        wandb=dict(
            entity="nupic-research",
            project="dendrite_baselines",
            name="spatial_pooler_50_2",
            group="spatial_pooler_50_2",
            notes="""
            Sparse network using random spatial pooler output as context.
            """
        )
    ),
)

SP_CONTEXT_100 = deepcopy(SP_CONTEXT_50)
SP_CONTEXT_100["dataset_args"].update(num_tasks=100)
SP_CONTEXT_100["env_config"]["wandb"].update(
    name="sp_context_100",
    group="sp_context_100",
)
SP_CONTEXT_100["model_args"].update(
    num_segments=100,
)
SP_CONTEXT_100.update(
    num_tasks=100,
    num_classes=10 * 100,
    num_samples=1,
)

SP_CONTEXT_100_NO_NORM = deepcopy(SP_CONTEXT_100)
SP_CONTEXT_100_NO_NORM["dataset_args"].update(normalize=False)
SP_CONTEXT_100_NO_NORM["env_config"]["wandb"].update(
    name="sp_context_100_no_norm",
    group="sp_context_100_no_norm",
)


# Search using 4 tasks only
SP_CONTEXT_4_SEARCH = deepcopy(SP_CONTEXT_10)
SP_CONTEXT_4_SEARCH["dataset_args"].update(num_tasks=4)
SP_CONTEXT_4_SEARCH["model_args"].update(num_segments=6)
SP_CONTEXT_4_SEARCH.update(
    epochs=2,
    batch_size=tune.sample_from(
        lambda spec: int(np.random.choice([64, 256]))),
    num_samples=20,
    num_tasks=4,
    num_classes=10 * 4,
    tasks_to_validate=(0, 1, 2, 3, 4),

    context_model_args=dict(
        kw_percent_on=tune.sample_from(
            lambda spec: np.random.choice([0.02, 0.05, 0.1, 0.15, 0.2])),
        boost_strength=tune.sample_from(
            lambda spec: np.random.choice([0.0, 0.5, 1.0, 2.0, 3.0])),
        weight_sparsity=tune.sample_from(
            lambda spec: np.random.choice([0.5, 0.75, 0.9])),
        duty_cycle_period=10240,
    ),

    optimizer_args=dict(
        lr=tune.sample_from(
            lambda spec: np.random.choice([0.001, 0.0001, 0.00001])),
    ),
)

# Export configurations in this file
CONFIGS = dict(
    sp_context_2=SP_CONTEXT_2,
    sp_context_4_search=SP_CONTEXT_4_SEARCH,
    sp_context_10=SP_CONTEXT_10,
    sp_context_50=SP_CONTEXT_50,
    sp_context_100=SP_CONTEXT_100,
    sp_context_100_no_norm=SP_CONTEXT_100_NO_NORM,
)
