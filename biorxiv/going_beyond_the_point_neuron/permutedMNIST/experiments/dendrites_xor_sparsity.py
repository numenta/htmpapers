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
Experiment file that runs networks which continually learn on permutedMNIST, but use
either active dendrites or sparse representations (via k-Winners), but not both. All
networks use the prototype method to infer context vectors at test time.
"""

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.continual_learning import mixins as cl_mixins
from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    DendriticMLP,
    ZeroSegmentDendriticLayer,
)
from nupic.research.frameworks.dendrites import mixins as dendrites_mixins
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins as vernon_mixins


class PrototypeExperiment(vernon_mixins.RezeroWeights,
                          dendrites_mixins.PrototypeContext,
                          cl_mixins.PermutedMNISTTaskIndices,
                          DendriteContinualLearningExperiment):
    pass


# ------------------------ CONFIGS FOR ACTIVE DENDRITES ONLY ------------------------ #

ACTIVE_DENDRITES_ONLY_BASE = dict(
    experiment_class=PrototypeExperiment,
    num_samples=8,

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
        output_size=10,
        hidden_sizes=[2048, 2048],
        dim_context=784,
        kw=False,
        kw_percent_on=0.0,  # `kw_percent_on` has no effect since `kw` is `False`
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
        dendritic_layer_class=AbsoluteMaxGatingDendriticLayer,
    ),

    batch_size=256,
    val_batch_size=512,
    tasks_to_validate=[1, 4, 9, 24, 49, 99],
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
)


ACTIVE_DENDRITES_ONLY_2 = deepcopy(ACTIVE_DENDRITES_ONLY_BASE)
ACTIVE_DENDRITES_ONLY_2["dataset_args"].update(num_tasks=2)
ACTIVE_DENDRITES_ONLY_2["model_args"].update(num_segments=2)
ACTIVE_DENDRITES_ONLY_2.update(
    num_tasks=2,
    num_classes=10 * 2,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=5e-4),
)


ACTIVE_DENDRITES_ONLY_5 = deepcopy(ACTIVE_DENDRITES_ONLY_BASE)
ACTIVE_DENDRITES_ONLY_5["dataset_args"].update(num_tasks=5)
ACTIVE_DENDRITES_ONLY_5["model_args"].update(num_segments=5)
ACTIVE_DENDRITES_ONLY_5.update(
    num_tasks=5,
    num_classes=10 * 5,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=10,
    optimizer_args=dict(lr=1e-3),
)


ACTIVE_DENDRITES_ONLY_10 = deepcopy(ACTIVE_DENDRITES_ONLY_BASE)
ACTIVE_DENDRITES_ONLY_10["dataset_args"].update(num_tasks=10)
ACTIVE_DENDRITES_ONLY_10["model_args"].update(num_segments=10)
ACTIVE_DENDRITES_ONLY_10.update(
    num_tasks=10,
    num_classes=10 * 10,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=5e-6),
)


ACTIVE_DENDRITES_ONLY_25 = deepcopy(ACTIVE_DENDRITES_ONLY_BASE)
ACTIVE_DENDRITES_ONLY_25["dataset_args"].update(num_tasks=25)
ACTIVE_DENDRITES_ONLY_25["model_args"].update(num_segments=25)
ACTIVE_DENDRITES_ONLY_25.update(
    num_tasks=25,
    num_classes=10 * 25,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=5e-6),
)


ACTIVE_DENDRITES_ONLY_50 = deepcopy(ACTIVE_DENDRITES_ONLY_BASE)
ACTIVE_DENDRITES_ONLY_50["dataset_args"].update(num_tasks=50)
ACTIVE_DENDRITES_ONLY_50["model_args"].update(num_segments=50)
ACTIVE_DENDRITES_ONLY_50.update(
    num_tasks=50,
    num_classes=10 * 50,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=1e-5),
)


ACTIVE_DENDRITES_ONLY_100 = deepcopy(ACTIVE_DENDRITES_ONLY_BASE)
ACTIVE_DENDRITES_ONLY_100["dataset_args"].update(num_tasks=100)
ACTIVE_DENDRITES_ONLY_100["model_args"].update(num_segments=100)
ACTIVE_DENDRITES_ONLY_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=5e-6),
)


# --------------------- CONFIGS FOR SPARSE REPRESENTATIONS ONLY --------------------- #

SPARSE_REPRESENTATIONS_ONLY_BASE = dict(
    experiment_class=PrototypeExperiment,
    num_samples=8,

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
        output_size=10,
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
        dendritic_layer_class=ZeroSegmentDendriticLayer,  # This dendritic layer is
                                                          # equivalent to a normal
                                                          # feed-forward layer
    ),

    batch_size=256,
    val_batch_size=512,
    tasks_to_validate=[1, 4, 9, 24, 49, 99],
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
)


SPARSE_REPRESENTATIONS_ONLY_2 = deepcopy(SPARSE_REPRESENTATIONS_ONLY_BASE)
SPARSE_REPRESENTATIONS_ONLY_2["dataset_args"].update(num_tasks=2)
SPARSE_REPRESENTATIONS_ONLY_2.update(
    num_tasks=2,
    num_classes=10 * 2,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=1e-3),
)


SPARSE_REPRESENTATIONS_ONLY_5 = deepcopy(SPARSE_REPRESENTATIONS_ONLY_BASE)
SPARSE_REPRESENTATIONS_ONLY_5["dataset_args"].update(num_tasks=5)
SPARSE_REPRESENTATIONS_ONLY_5.update(
    num_tasks=5,
    num_classes=10 * 5,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=10,
    optimizer_args=dict(lr=5e-5),
)


SPARSE_REPRESENTATIONS_ONLY_10 = deepcopy(SPARSE_REPRESENTATIONS_ONLY_BASE)
SPARSE_REPRESENTATIONS_ONLY_10["dataset_args"].update(num_tasks=10)
SPARSE_REPRESENTATIONS_ONLY_10.update(
    num_tasks=10,
    num_classes=10 * 10,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=1e-4),
)


SPARSE_REPRESENTATIONS_ONLY_25 = deepcopy(SPARSE_REPRESENTATIONS_ONLY_BASE)
SPARSE_REPRESENTATIONS_ONLY_25["dataset_args"].update(num_tasks=25)
SPARSE_REPRESENTATIONS_ONLY_25.update(
    num_tasks=25,
    num_classes=10 * 25,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=10,
    optimizer_args=dict(lr=1e-5),
)


SPARSE_REPRESENTATIONS_ONLY_50 = deepcopy(SPARSE_REPRESENTATIONS_ONLY_BASE)
SPARSE_REPRESENTATIONS_ONLY_50["dataset_args"].update(num_tasks=50)
SPARSE_REPRESENTATIONS_ONLY_50.update(
    num_tasks=50,
    num_classes=10 * 50,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=10,
    optimizer_args=dict(lr=1e-5),
)


SPARSE_REPRESENTATIONS_ONLY_100 = deepcopy(SPARSE_REPRESENTATIONS_ONLY_BASE)
SPARSE_REPRESENTATIONS_ONLY_100["dataset_args"].update(num_tasks=100)
SPARSE_REPRESENTATIONS_ONLY_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=1e-4),
)


# ----------------------------------------------------------------------------------- #

CONFIGS = dict(
    active_dendrites_only_2=ACTIVE_DENDRITES_ONLY_2,
    active_dendrites_only_5=ACTIVE_DENDRITES_ONLY_5,
    active_dendrites_only_10=ACTIVE_DENDRITES_ONLY_10,
    active_dendrites_only_25=ACTIVE_DENDRITES_ONLY_25,
    active_dendrites_only_50=ACTIVE_DENDRITES_ONLY_50,
    active_dendrites_only_100=ACTIVE_DENDRITES_ONLY_100,
    sparse_representations_only_2=SPARSE_REPRESENTATIONS_ONLY_2,
    sparse_representations_only_5=SPARSE_REPRESENTATIONS_ONLY_5,
    sparse_representations_only_10=SPARSE_REPRESENTATIONS_ONLY_10,
    sparse_representations_only_25=SPARSE_REPRESENTATIONS_ONLY_25,
    sparse_representations_only_50=SPARSE_REPRESENTATIONS_ONLY_50,
    sparse_representations_only_100=SPARSE_REPRESENTATIONS_ONLY_100,
)
