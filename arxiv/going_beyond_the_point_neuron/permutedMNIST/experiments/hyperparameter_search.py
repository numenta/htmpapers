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
Experiment file to run hyperparameter searches on different types of dendritic
networks in a continual learning setting using permutedMNIST as benchmark.
This config file was used to generate the data in CNS 2021's poster and
Bernstein Conference 2021 submission.
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

"""Permuted MNIST with DendriticMLP"""


class NbSegmentSearchExperiment(
    mixins.RezeroWeights,
    mixins.CentroidContext,
    mixins.PermutedMNISTTaskIndices,
    DendriteContinualLearningExperiment,
):
    pass


BASE10 = dict(
    experiment_class=NbSegmentSearchExperiment,
    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites/"),
    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=10,
        # Consistent location outside of git repo
        root=os.path.expanduser("~/nta/results/data/"),
        seed=42,
        download=False,  # Change to True if running for the first time
    ),
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[2048, 2048],
        num_segments=tune.grid_search([2, 3, 5, 7, 10, 14, 20, 30, 50, 100]),
        dim_context=784,
        kw=True,
        kw_percent_on=tune.grid_search(
            [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
        ),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.grid_search(
            [0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        ),
        context_percent_on=0.1,
    ),
    batch_size=256,
    val_batch_size=512,
    epochs=3,
    num_tasks=10,
    tasks_to_validate=range(10),  # Tasks on which to run validate
    num_classes=10 * 10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=10,
    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=5e-4),
)

# varying only num segments
SEGMENT_SEARCH = deepcopy(BASE10)
SEGMENT_SEARCH["model_args"].update(kw_percent_on=0.1, weight_sparsity=0.5)

# varying only kw sparsity
KW_SPARSITY_SEARCH = deepcopy(BASE10)
KW_SPARSITY_SEARCH["model_args"].update(num_segments=10, weight_sparsity=0.5)

# varying only weights sparsity
W_SPARSITY_SEARCH = deepcopy(BASE10)
W_SPARSITY_SEARCH["model_args"].update(num_segments=10, kw_percent_on=0.1)

# test config for 10 tasks
TEST = deepcopy(BASE10)
TEST["model_args"].update(
    kw_percent_on=0.1, weight_sparsity=0.5, num_segments=2
)
TEST["num_samples"] = 1

# Idem on 50 tasks
BASE50 = deepcopy(BASE10)
BASE50["dataset_args"].update(num_tasks=50)
BASE50["tasks_to_validate"] = range(50)
BASE50["num_classes"] = 10 * 50

# Segment search
SEGMENT_SEARCH_50 = deepcopy(BASE50)
SEGMENT_SEARCH_50["model_args"].update(kw_percent_on=0.1, weight_sparsity=0.5)

# kw sparsity search
KW_SPARSITY_SEARCH_50 = deepcopy(BASE50)
KW_SPARSITY_SEARCH_50["model_args"].update(
    num_segments=50, weight_sparsity=0.5
)

# weight sparsity search
W_SPARSITY_SEARCH_50 = deepcopy(BASE50)
W_SPARSITY_SEARCH_50["model_args"].update(num_segments=50, kw_percent_on=0.1)

# test config for 50 tasks
TEST50 = deepcopy(BASE50)
TEST50["model_args"].update(
    kw_percent_on=0.1, weight_sparsity=0.5, num_segments=2
)
TEST50["num_samples"] = 1

# optimal model
OPTIMAL_50 = deepcopy(BASE50)
OPTIMAL_50["model_args"].update(
    kw_percent_on=0.05, weight_sparsity=0.5, num_segments=100
)


# CROSS HYPERPARAMETERS SEARCH #
# cross hyperparameter search for 10 tasks
CROSS_SEARCH = deepcopy(BASE10)
CROSS_SEARCH["model_args"].update(
    num_segments=tune.grid_search([5, 10, 50, 100]),
    kw_percent_on=tune.grid_search([0.05, 0.2, 0.6, 0.9, 0.99]),
    weight_sparsity=tune.grid_search([0.05, 0.5, 0.9]),
)
CROSS_SEARCH["optimizer_args"] = dict(lr=tune.grid_search([5e-4, 5e-3, 5e-2]))
CROSS_SEARCH["num_samples"] = 5

# Export configurations in this file
CONFIGS = dict(
    segment_search=SEGMENT_SEARCH,
    kw_sparsity_search=KW_SPARSITY_SEARCH,
    w_sparsity_search=W_SPARSITY_SEARCH,
    segment_search_50=SEGMENT_SEARCH_50,
    kw_sparsity_search_50=KW_SPARSITY_SEARCH_50,
    w_sparsity_search_50=W_SPARSITY_SEARCH_50,
    test=TEST,
    test50=TEST50,
    optimal_50=OPTIMAL_50,
    cross_search=CROSS_SEARCH,
)
