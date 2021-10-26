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
import torch.nn as nn

from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.dendrites.mixins import SpatialPoolerAnalysis
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins
from nupic.torch.modules import KWinners, SparseWeights


class SPExperiment(mixins.RezeroWeights,
                   mixins.UpdateBoostStrength,
                   mixins.PermutedMNISTTaskIndices,
                   SpatialPoolerAnalysis,
                   DendriteContinualLearningExperiment):
    pass


class SP(nn.Module):
    """
    A simple spatial pooler like network to be used as a context vector.

    :param input_size: size of the input to the network
    :param output_size: the number of units in the output layer
    :param kw_percent_on: percent of hidden units activated by K-winners. If 0, use ReLU
    :param boost_strength:
    :param weight_sparsity: the sparsity level of feed-forward weights.
    :param duty_cycle_period:
    """

    def __init__(
        self, input_size, output_size, kw_percent_on=0.05, boost_strength=0.0,
        weight_sparsity=0.95, duty_cycle_period=1000,
    ):
        super().__init__()

        self.linear = SparseWeights(nn.Linear(input_size, output_size),
                                    sparsity=weight_sparsity,
                                    allow_extremes=True)
        self.kw = KWinners(n=output_size, percent_on=kw_percent_on,
                           boost_strength=boost_strength,
                           duty_cycle_period=duty_cycle_period)

    def forward(self, x):
        return self.kw(self.linear(x))


# Centroid method for inferring contexts: 50 permutedMNIST tasks
SP_PROTOTYPE_50 = dict(
    experiment_class=SPExperiment,
    num_samples=1,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=50,
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        seed=42,
    ),

    model_class=SP,
    model_args=dict(
        input_size=784,
        output_size=500,
        kw_percent_on=0.05,
        boost_strength=0.0,
        weight_sparsity=0.9,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=1,
    tasks_to_validate=[0],
    num_tasks=50,
    num_classes=10 * 50,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    optimizer_args=dict(lr=0.0),
    distance_metric="cosine",
)

# For debugging
SP_PROTOTYPE_2 = deepcopy(SP_PROTOTYPE_50)
SP_PROTOTYPE_2["dataset_args"].update(num_tasks=2)
SP_PROTOTYPE_2.update(
    num_tasks=2,
    num_classes=10 * 2,
)

SP_PROTOTYPE_10 = deepcopy(SP_PROTOTYPE_50)
SP_PROTOTYPE_10["dataset_args"].update(num_tasks=10)
SP_PROTOTYPE_10.update(
    num_tasks=10,
    num_classes=10 * 10,
    model_args=dict(
        input_size=784,
        output_size=500,
        kw_percent_on=0.02,
        boost_strength=0.0,
        weight_sparsity=0.75,
    ),
)

SP10_SEARCH = deepcopy(SP_PROTOTYPE_10)
SP10_SEARCH.update(
    num_samples=20,
    model_args=dict(
        input_size=784,
        output_size=tune.sample_from(
            lambda spec: np.random.choice([250, 500])),
        kw_percent_on=tune.sample_from(
            lambda spec: np.random.choice([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])),
        boost_strength=tune.sample_from(
            lambda spec: np.random.choice([0.0, 0.5, 1.0, 2.0])),
        weight_sparsity=tune.sample_from(
            lambda spec: np.random.choice([0.5, 0.75, 0.9])),
        duty_cycle_period=tune.sample_from(
            lambda spec: np.random.choice([1000, 10240])),
    ),

)

# Export configurations in this file
CONFIGS = dict(
    sp_proto50=SP_PROTOTYPE_50,
    sp_proto10=SP_PROTOTYPE_10,
    sp_proto2=SP_PROTOTYPE_2,
    sp10_search=SP10_SEARCH,
)
