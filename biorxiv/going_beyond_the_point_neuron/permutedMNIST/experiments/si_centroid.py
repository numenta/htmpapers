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
Experiment file that runs dendritic networks where (a) the context vector is inferred
during inference via prototyping, and (b) Synaptic Intelligence (SI) is applied to the
feed-forward parameters.
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
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins


class SICentroidExperiment(mixins.SynapticIntelligence,
                           mixins.RezeroWeights,
                           mixins.CentroidContext,
                           mixins.PermutedMNISTTaskIndices,
                           DendriteContinualLearningExperiment):
    pass


# Synaptic Intelligence + Dendrites on 10 permutedMNIST tasks
SI_CENTROID_10 = dict(
    experiment_class=SICentroidExperiment,
    num_samples=8,

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
        output_size=10,
        hidden_sizes=[2000, 2000],  # Note we use 2000 hidden units instead of 2048 for
                                    # a better comparison with SI and XdG
        num_segments=10,
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    si_args=dict(
        c=0.1,
        damping=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=20,  # Note that SI only works well with ~20 epochs of training per task
    tasks_to_validate=[0, 1, 2, 3, 4, 9, 24, 49, 74, 99],
    num_tasks=10,
    num_classes=10 * 10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
    optimizer_args=dict(lr=5e-4),
    reset_optimizer_after_task=False,  # The SI paper reports not resetting the Adam
                                       # optimizer between tasks, and this
                                       # works well with dendrites too
)

# Synaptic Intelligence + Dendrites on 50 permutedMNIST tasks
SI_CENTROID_50 = deepcopy(SI_CENTROID_10)
SI_CENTROID_50["dataset_args"].update(num_tasks=50)
SI_CENTROID_50["model_args"].update(num_segments=50)
SI_CENTROID_50.update(
    num_tasks=50,
    num_classes=10 * 50,
)

SI_CENTROID_100 = deepcopy(SI_CENTROID_50)
SI_CENTROID_100["dataset_args"].update(num_tasks=100)
SI_CENTROID_100["model_args"].update(num_segments=100)
SI_CENTROID_100.update(
    num_tasks=100,
    num_classes=10 * 100,
    optimizer_args=dict(lr=5e-4),
)


# -------------------------- HYPERPARAMETER SEARCH CONFIGS -------------------------- #

# HP search on dendrites for 10 tasks with SI FF + SI Dendrites.
SI_CENTROID_HP_10 = deepcopy(SI_CENTROID_10)
SI_CENTROID_HP_10["model_args"].update(
    kw_percent_on=0.1,
    weight_sparsity=0.5,
    num_segments=tune.grid_search([2, 3, 5, 7, 10, 14, 20, 30, 50, 100])
)
SI_CENTROID_HP_10.update(
    tasks_to_validate=[9],

    si_args=dict(
        c=0.1,
        damping=0.1,
        apply_to_dendrites=True
    )
)

# HP search on dendrites for 10 tasks with SI FF.
SI_CENTROID_HP_10_CONTROL = deepcopy(SI_CENTROID_HP_10)
SI_CENTROID_HP_10_CONTROL["model_args"].update(
    kw_percent_on=0.1,
    weight_sparsity=0.5,
    num_segments=tune.grid_search([2, 3, 5, 7, 10, 14, 20, 30, 50, 100])
)
SI_CENTROID_HP_10_CONTROL.update(
    tasks_to_validate=[10],

    si_args=dict(
        c=0.1,
        damping=0.1,
        apply_to_dendrites=False
    )
)

# Test on 50 tasks with SI FF + SI Dendrites.
SI_CENTROID_50_TEST_WITH_SI = deepcopy(SI_CENTROID_50)
SI_CENTROID_50_TEST_WITH_SI["model_args"].update(
    kw_percent_on=0.1,
    weight_sparsity=0.5,
    num_segments=50
)
SI_CENTROID_50_TEST_WITH_SI.update(
    num_samples=8,
    tasks_to_validate=[49],

    si_args=dict(
        c=0.1,
        damping=0.1,
        apply_to_dendrites=True
    )
)

# Export configurations in this file
CONFIGS = dict(
    si_centroid_10=SI_CENTROID_10,
    si_centroid_50=SI_CENTROID_50,
    si_centroid_100=SI_CENTROID_100,
    si_centroid_hp_10=SI_CENTROID_HP_10,
    si_centroid_hp_10_control=SI_CENTROID_HP_10_CONTROL,
    si_centroid_50_test_with_si=SI_CENTROID_50_TEST_WITH_SI
)
