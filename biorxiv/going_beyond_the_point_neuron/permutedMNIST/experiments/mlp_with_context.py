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
Experiments designed to investigate different dendritic functions that mix feedforward
and dendritic inputs. Examples include additive bias, multiplicative, multiplicative
gating, etc.
"""

from copy import deepcopy

import numpy as np
import ray.tune as tune

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.modules.dendritic_layers import (
    ZeroSegmentDendriticLayer,
)
from nupic.research.frameworks.pytorch.datasets import ContextDependentPermutedMNIST
from nupic.research.frameworks.pytorch.models import ModifiedInitStandardMLP, SparseMLP

from .mlp import THREE_LAYER_MLP_10
from .no_dendrites import NoDendriteExperiment

# 10 tasks, dense MLP, onehot context vector concatenated to the input
THREE_LAYER_MLP_10_ONEHOT = deepcopy(THREE_LAYER_MLP_10)
THREE_LAYER_MLP_10_ONEHOT.update(
    dataset_class=ContextDependentPermutedMNIST,
    dataset_args=dict(
        num_tasks=10,
        download=True,
        seed=np.random.randint(2, 10_000),
        context_type="one_hot",
        combine_context_as="concatenate",
    ),
    num_samples=1,
    num_tasks=10,
    num_classes=10 * 10,
    model_class=ModifiedInitStandardMLP,
    model_args=dict(
        input_size=784 + 10,  # + 10 due to 10 tasks
        hidden_sizes=[2048, 2048],
        num_classes=10 * 10,
    ),

    optimizer_args=dict(lr=tune.grid_search(
        [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])),
    # For wandb
    env_config=dict(
        wandb=dict(
            entity="nupic-research",
            project="dendrite_baselines",
            name="THREE_LAYER_MLP_10_ONEHOT",
            group="MLP_ABLATIONS",
        ),
    ),
)

# 10 tasks, dense mlp, prototype (mean image per task) as context concatenated
# to the input
THREE_LAYER_MLP_10_PROTOTYPE = deepcopy(THREE_LAYER_MLP_10_ONEHOT)
THREE_LAYER_MLP_10_PROTOTYPE["dataset_args"].update(
    context_type="prototype",
    combine_context_as="concatenate",
)
THREE_LAYER_MLP_10_PROTOTYPE["model_args"].update(
    input_size=784 + 784,  # 784 image + 784 context
)

# 10 tasks, one random sparse binary vector as context per task concatenated
# to the input
THREE_LAYER_MLP_10_SPARSE_BINARY = deepcopy(THREE_LAYER_MLP_10_ONEHOT)
THREE_LAYER_MLP_10_SPARSE_BINARY["dataset_args"].update(
    context_type="sparse_binary",
    combine_context_as="concatenate",
    dim_context=784,
)
THREE_LAYER_MLP_10_SPARSE_BINARY["model_args"].update(
    input_size=784 + 784
)

# As above, but sparse weights. Note that configs that say three_layer_MLP
# are no longer used if there is sparsity, as these include things like boosting
# and duty cycles. I kept these configs here as more of a historical record.
THREE_LAYER_MLP_10_ONEHOT_SPARSE = deepcopy(THREE_LAYER_MLP_10_ONEHOT)
THREE_LAYER_MLP_10_ONEHOT_SPARSE.update(
    model_class=SparseMLP,
    model_args=dict(
        kw_percent_on=(1., 1.),
        weight_sparsity=tune.grid_search([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]),
        input_size=784 + 10,  # + 10 due to 10 tasks
        hidden_sizes=[2048, 2048],
        output_size=10 * 10,
    )
)

THREE_LAYER_MLP_10_PROTOTYPE_SPARSE = deepcopy(THREE_LAYER_MLP_10_PROTOTYPE)
THREE_LAYER_MLP_10_PROTOTYPE_SPARSE.update(
    model_class=SparseMLP,
    model_args=dict(
        kw_percent_on=(1., 1.),
        weight_sparsity=tune.grid_search([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]),
        input_size=784 + 784,  # + 10 due to 10 tasks
        hidden_sizes=[2048, 2048],
        output_size=10 * 10,
    )
)
THREE_LAYER_MLP_10_PROTOTYPE_SPARSE["model_args"].update(
    weight_sparsity=tune.grid_search([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]),
)
THREE_LAYER_MLP_10_PROTOTYPE_SPARSE["env_config"]["wandb"].update(
    name="THREE_LAYER_MLP_10_PROTOTYPE_SPARSE",
)

THREE_LAYER_MLP_10_ONEHOT_DENSE_KW = deepcopy(THREE_LAYER_MLP_10_ONEHOT_SPARSE)
THREE_LAYER_MLP_10_ONEHOT_DENSE_KW["model_args"].update(
    weight_sparsity=(0., 0.),
    kw_percent_on=tune.grid_search(
        [(.01, 0.1), (.05, .05), (.1, .1), (.25, .25), (.5, .5)])
)

THREE_LAYER_MLP_10_PROTOTYPE_DENSE_KW = deepcopy(THREE_LAYER_MLP_10_PROTOTYPE_SPARSE)
THREE_LAYER_MLP_10_PROTOTYPE_DENSE_KW["model_args"].update(
    weight_sparsity=(0., 0.),
    kw_percent_on=tune.grid_search(
        [(.01, 0.1), (.05, .05), (.1, .1), (.25, .25), (.5, .5)])
)

THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW = deepcopy(THREE_LAYER_MLP_10_ONEHOT_SPARSE)
THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW["model_args"].update(
    kw_percent_on=tune.grid_search(
        [(.01, 0.1), (.05, .05), (.1, .1), (.25, .25), (.5, .5)]),
)

THREE_LAYER_MLP_10_PROTOTYPE_SPARSE_KW = deepcopy(THREE_LAYER_MLP_10_PROTOTYPE_SPARSE)
THREE_LAYER_MLP_10_PROTOTYPE_SPARSE_KW["model_args"].update(
    kw_percent_on=tune.grid_search(
        [(.01, 0.1), (.05, .05), (.1, .1), (.25, .25), (.5, .5)]),
)

###
# Zero segment instead of SparseMLP
###

# 10 tasks, onehot vector with task id concatenated to input
# k winners activations, but dense weights
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW_ = deepcopy(
    THREE_LAYER_MLP_10_ONEHOT_DENSE_KW)
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW_.update(
    experiment_class=NoDendriteExperiment,
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784 + 10,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    )
)

# as above, but prototype context concatenated to input image
THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_ = deepcopy(
    THREE_LAYER_MLP_10_PROTOTYPE_DENSE_KW)
THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_.update(
    experiment_class=NoDendriteExperiment,
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784 + 784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    )
)


# 10 tasks, onehot task vector concatenated to input
# sparse weights, dense activations
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_ = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_.update(
    model_args=dict(
        input_size=784 + 10,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=False,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    )
)

# as above, but prototype context
THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_SPARSE_ = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_SPARSE_.update(
    model_args=dict(
        input_size=784 + 784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=False,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    )
)

# as above, onehot context vector, BOTH sparse weights and activations
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_KW_ = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_KW_["model_args"].update(
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9])
)

THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_SPARSE_KW_ = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_SPARSE_KW_["model_args"].update(
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9])
)

###
# Scan n tasks
# Pick the best MLP from all the models above, and scan up to 30, 50, 100 tasks
# Best model happened to use prototype context, lr=.0001, kw_percent_on = .05
# Hyperparameters were fixed for some of these experiments. The suffix scan_epoch means
# for each num_tasks (03, 50 , 100), tune the number of epochs to train for per task.
# The suffix best_epoch means using the number of epochs found from scan_epochs.
###

# 30 tasks, no epoch tuning
THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_ = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_.update(
    num_samples=8,
    dataset_args=dict(
        num_tasks=30,
        download=True,
        seed=np.random.randint(2, 10_000),
        context_type="prototype",
        combine_context_as="concatenate",
    ),
    model_args=dict(
        input_size=784 + 784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),
    num_classes=10 * 30,
    num_tasks=30,
    tasks_to_validate=[29],
    optimizer_args=dict(lr=0.0001),
)


# 30 tasks, tune the number of epochs
THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_SCAN_EPOCH = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_
)
THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_SCAN_EPOCH.update(
    epochs=tune.grid_search([1, 2, 3, 4]),
    num_samples=1,
)

# 30 tasks, using the tuned epochs parameter. Runs 8 times so you can average over
# different random seeds.
THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_BEST_EPOCH = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_
)
THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_BEST_EPOCH.update(
    epochs=3,
)

# Did not scan epochs for the 10 task model, just use best hyperparameters
# and average over 8 runs.
THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_AVG = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_AVG.update(
    num_classes=10 * 10,
    num_tasks=10,
    tasks_to_validate=[9],
)
THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_AVG["dataset_args"].update(
    num_tasks=10
)

# 50 tasks, prior to tuning number of epochs
THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_ = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_.update(
    num_classes=10 * 50,
    num_tasks=50,
    tasks_to_validate=[49],
)
THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_["dataset_args"].update(
    num_tasks=50
)

# 50 tasks, scan the number of epochs
THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_SCAN_EPOCH = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_
)
THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_SCAN_EPOCH.update(
    epochs=tune.grid_search([1, 2, 3, 4]),
    num_samples=1,
)

# 50 tasks, use best number of epochs
THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_BEST_EPOCH = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_
)
THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_BEST_EPOCH.update(
    epochs=2,
)

# 100 tasks, pre epoch tuning
THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_ = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_.update(
    num_classes=10 * 100,
    num_tasks=100,
    tasks_to_validate=[99],
)
THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_["dataset_args"].update(
    num_tasks=100
)

# 100 tasks, tuning epochs
THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_SCAN_EPOCH = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_SCAN_EPOCH.update(
    epochs=tune.grid_search([1, 2, 3, 4]),
    num_samples=1,
)

# 100 tasks, best epoch parameter
THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_BEST_EPOCH = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_
)
THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_BEST_EPOCH.update(
    epochs=1,
)


# 250 tasks, pre epoch tuning. I never actually ran this, but results were suggesting
# that the performance gap between best MLP with context, and dendrites model is widest
# when the number of tasks is largest. Running with way more tasks could be interesting
# to try, so the config is available.
THREE_LAYER_ZERO_SEGMENT_250_PROTOTYPE_DENSE_KW_ = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_250_PROTOTYPE_DENSE_KW_.update(
    num_classes=10 * 250,
    num_tasks=250,
    tasks_to_validate=[249],
)
THREE_LAYER_ZERO_SEGMENT_250_PROTOTYPE_DENSE_KW_["dataset_args"].update(
    num_tasks=250
)

###
# Hyperparameter tuning for MLP or ZeroSegmentDendrite model using sparse binary context
###

THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_SPARSE = deepcopy(
    THREE_LAYER_MLP_10_SPARSE_BINARY)
THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_SPARSE.update(
    experiment_class=NoDendriteExperiment,
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784 + 784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=False,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),
)

THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_DENSE_KW = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_SPARSE)
THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_DENSE_KW.update(
    model_args=dict(
        input_size=784 + 784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),
)

THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_SPARSE_KW = deepcopy(
    THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_DENSE_KW)
THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_SPARSE_KW["model_args"].update(
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
)

CONFIGS = dict(
    # onehot context mlp
    three_layer_mlp_10_onehot=THREE_LAYER_MLP_10_ONEHOT,
    three_layer_mlp_10_onehot_sparse=THREE_LAYER_MLP_10_ONEHOT_SPARSE,
    three_layer_mlp_10_onehot_dense_kw=THREE_LAYER_MLP_10_ONEHOT_DENSE_KW,
    three_layer_mlp_10_onehot_sparse_kw=THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW,

    # prototype context mlp
    three_layer_mlp_10_prototype=THREE_LAYER_MLP_10_PROTOTYPE,
    three_layer_mlp_10_prototype_sparse=THREE_LAYER_MLP_10_PROTOTYPE_SPARSE,
    three_layer_mlp_10_prototype_dense_kw=THREE_LAYER_MLP_10_PROTOTYPE_DENSE_KW,
    three_layer_mlp_10_prototype_sparse_kw=THREE_LAYER_MLP_10_PROTOTYPE_SPARSE_KW,

    # sparse binary mlp
    three_layer_mlp_10_sparse_binary=THREE_LAYER_MLP_10_SPARSE_BINARY,

    # Zero segment onehot context
    three_layer_zero_segment_10_onehot_dense_kw_=THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW_,  # noqa E501
    three_layer_zero_segment_10_onehot_sparse_=THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_,  # noqa E501
    three_layer_zero_segment_10_onehot_sparse_kw_=THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_KW_,  # noqa E501

    # Zero segment prototype context
    three_layer_zero_segment_10_prototype_dense_kw_=THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_,  # noqa E501
    three_layer_zero_segment_10_prototype_sparse_=THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_SPARSE_,  # noqa E501
    three_layer_zero_segment_10_prototype_sparse_kw_=THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_SPARSE_KW_,  # noqa E501

    # Zero segment sparse binary
    three_layer_zero_segment_10_sparse_binary_sparse=THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_SPARSE,  # noqa E501
    three_layer_zero_segment_10_sparse_binary_dense_kw=THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_DENSE_KW,  # noqa E501
    three_layer_zero_segment_10_sparse_binary_sparse_kw=THREE_LAYER_ZERO_SEGMENT_10_SPARSE_BINARY_SPARSE_KW,  # noqa E501

    # Scan number of tasks
    three_layer_zero_segment_10_prototype_dense_kw___=THREE_LAYER_ZERO_SEGMENT_10_PROTOTYPE_DENSE_KW_AVG,  # noqa E501
    three_layer_zero_segment_30_prototype_dense_kw___=THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_,  # noqa E501
    three_layer_zero_segment_50_prototype_dense_kw___=THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_,  # noqa E501
    three_layer_zero_segment_100_prototype_dense_kw___=THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_,  # noqa E501

    # Check num epochs for each num_tasks in best model
    three_layer_zero_segment_30_prototype_dense_kw_scan_epoch=THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_SCAN_EPOCH,  # noqa E501
    three_layer_zero_segment_50_prototype_dense_kw_scan_epoch=THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_SCAN_EPOCH,  # noqa E501
    three_layer_zero_segment_100_prototype_dense_kw_scan_epoch=THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_SCAN_EPOCH,  # noqa E501

    # Scan number of tasks using optimal n_epochs
    three_layer_zero_segment_30_prototype_dense_kw_best_epoch=THREE_LAYER_ZERO_SEGMENT_30_PROTOTYPE_DENSE_KW_BEST_EPOCH,  # noqa E501
    three_layer_zero_segment_50_prototype_dense_kw_best_epoch=THREE_LAYER_ZERO_SEGMENT_50_PROTOTYPE_DENSE_KW_BEST_EPOCH,  # noqa E501
    three_layer_zero_segment_100_prototype_dense_kw_best_epoch=THREE_LAYER_ZERO_SEGMENT_100_PROTOTYPE_DENSE_KW_BEST_EPOCH,  # noqa E501
)
