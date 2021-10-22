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
Regular batch MNIST with standard sparse and dense networks. Used as sanity check
and some parameter tuning.
"""

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.models import StandardMLP
from nupic.research.frameworks.pytorch.models.common_models import SparseMLP
from nupic.research.frameworks.vernon import SupervisedExperiment, mixins


class SparseMLPExperiment(mixins.RezeroWeights,
                          mixins.UpdateBoostStrength,
                          SupervisedExperiment):
    pass


# Gets about 98% accuracy
DENSE_MLP = dict(
    experiment_class=SupervisedExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=datasets.MNIST,
    dataset_args=dict(
        # Consistent location outside of git repo
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.13062755,), (0.30810780,)),
        ])
    ),

    model_class=StandardMLP,
    model_args=dict(
        input_size=784,
        num_classes=10,
        hidden_sizes=[2048, 2048],
    ),

    batch_size=32,
    val_batch_size=512,
    epochs=13,
    epochs_to_validate=range(30),
    num_classes=10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=1,  # Increase to run multiple experiments in parallel

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.SGD,
    optimizer_args=dict(
        lr=0.01,
    ),

    # Learning rate scheduler class and args. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    lr_scheduler_args=dict(
        gamma=0.1,
        step_size=10,
    ),
)

# Default parameters with 0.05 percent sparsities, got about 90.5%
# This version gets about 97%. It's possible to go up to 97.4% if we
# use StepLR.
SPARSE_MLP = dict(
    experiment_class=SparseMLPExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=datasets.MNIST,
    dataset_args=dict(
        # Consistent location outside of git repo
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.13062755,), (0.30810780,)),
        ])
    ),

    model_class=SparseMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(0.05, 0.05),
        weight_sparsity=(0.05, 0.05),
        boost_strength=0.0,
        boost_strength_factor=0.0,
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),

    batch_size=128,
    val_batch_size=512,
    epochs=20,
    epochs_to_validate=range(30),
    num_classes=10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=3,  # Increase to run multiple experiments in parallel

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.SGD,
    optimizer_args=dict(
        lr=0.1,
        momentum=0.9,
    ),

)

# This version is slightly denser and gets about 98.4%
SPARSE_MLP15 = deepcopy(SPARSE_MLP)
SPARSE_MLP15.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(0.15, 0.15),
        weight_sparsity=(0.2, 0.2),
        boost_strength=0.0,
        boost_strength_factor=0.0,
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),

    epochs=15,
    optimizer_args=dict(
        lr=0.1,
        momentum=0.9,
    ),
)

# Sparse weights only
#  5% = 92.81%
# 10% = 94.27%
# 15% = 95.46%
SPARSE_WTS_MLP = deepcopy(SPARSE_MLP)
SPARSE_WTS_MLP.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(1.0, 1.0),
        weight_sparsity=(0.05, 0.05),
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),
)

# Sparse activations only
#  5% = 96.7%
# 10% = 97.16
# 15% = 97.33%
SPARSE_ACT_MLP = deepcopy(SPARSE_MLP)
SPARSE_ACT_MLP.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(0.15, 0.15),
        # kw_percent_on=tune.grid_search([(0.05, 0.05), (0.1, 0.1), (0.15, 0.15), ]),
        weight_sparsity=(1.0, 1.0),
        boost_strength=1.5,
        boost_strength_factor=0.8,
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),
)

SPARSITY_SEARCH = deepcopy(SPARSE_MLP)
SPARSITY_SEARCH.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=tune.grid_search([(0.05, 0.05), (0.1, 0.1), (0.15, 0.15),
                                        (0.2, 0.2), ]),
        weight_sparsity=tune.grid_search([(0.05, 0.05), (0.1, 0.1), (0.15, 0.15),
                                          (0.2, 0.2), ]),
        boost_strength=0.0,
        boost_strength_factor=0.0,
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),
)

SPARSITY_SEARCH2 = deepcopy(SPARSE_MLP)
SPARSITY_SEARCH2.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(0.1, 0.1),
        weight_sparsity=tune.grid_search([(0.0, 0.0), (0.25, 0.25), (0.35, 0.35), ]),
        boost_strength=0.0,
        boost_strength_factor=0.0,
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),
)

SPARSITY_SEARCH3 = deepcopy(SPARSE_MLP)
SPARSITY_SEARCH3.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(0.15, 0.15),
        weight_sparsity=(0.2, 0.2),
        boost_strength=tune.sample_from(lambda spec: 1.5 * np.random.random()),
        boost_strength_factor=tune.sample_from(lambda spec: np.random.random()),
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),

    batch_size=tune.sample_from(lambda spec: int(np.random.choice([32, 64, 128]))),
    epochs=25,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=100,  # Increase to run multiple experiments in parallel

    optimizer_args=dict(
        lr=tune.sample_from(lambda spec: np.random.choice([0.1, 0.01, 0.001])),
        momentum=tune.sample_from(lambda spec: np.random.choice([0.0, 0.5, 0.9])),
    ),

    # Learning rate scheduler class and args. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    lr_scheduler_args=dict(
        gamma=tune.sample_from(lambda spec: 0.99 * np.random.random()),
        step_size=tune.sample_from(lambda spec: np.random.randint(1, 10)),
    ),
)

SPARSITY_SEARCH4 = deepcopy(SPARSE_MLP)
SPARSITY_SEARCH4.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(0.15, 0.15),
        weight_sparsity=(0.2, 0.2),
        boost_strength=tune.sample_from(lambda spec: 0.5 * np.random.random()),
        boost_strength_factor=tune.sample_from(lambda spec: np.random.random()),
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),

    batch_size=128,
    epochs=25,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=20,

    optimizer_args=dict(
        lr=0.1,
        momentum=0.9,
    ),

    # Learning rate scheduler class and args. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    lr_scheduler_args=dict(
        gamma=tune.sample_from(lambda spec: 0.5 * np.random.random()),
        step_size=5,
    ),
)

# Gets to about 97.7% accuracy but requires step LR and high LR which might not
# work for CL.
SPARSITY_SEARCH5 = deepcopy(SPARSE_MLP)
SPARSITY_SEARCH5.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(0.05, 0.05),
        weight_sparsity=(0.05, 0.05),
        boost_strength=tune.sample_from(lambda spec: 0.5 * np.random.random()),
        boost_strength_factor=tune.sample_from(lambda spec: np.random.random()),
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),

    batch_size=128,
    epochs=15,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=30,  # Increase to run multiple experiments in parallel

    optimizer_args=dict(
        lr=tune.sample_from(lambda spec: np.random.choice([0.1, 0.01])),
        momentum=tune.sample_from(lambda spec: np.random.choice([0.9, 0.95, 0.975])),
    ),

    # Learning rate scheduler class and args. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    lr_scheduler_args=dict(
        gamma=tune.sample_from(lambda spec: 0.99 * np.random.random()),
        step_size=tune.sample_from(lambda spec: np.random.randint(1, 10)),
    ),
)

# Trying without StepLR or boosting
SPARSITY_SEARCH6 = deepcopy(SPARSE_MLP)
SPARSITY_SEARCH6.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        kw_percent_on=(0.05, 0.05),
        weight_sparsity=(0.05, 0.05),
        boost_strength=0.0,
        boost_strength_factor=0.0,
        k_inference_factor=1.0,
        use_batch_norm=False,
        hidden_sizes=(2048, 2048),
    ),

    batch_size=128,
    epochs=25,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=40,  # Increase to run multiple experiments in parallel

    optimizer_args=dict(
        lr=tune.sample_from(lambda spec: np.random.uniform(0.001, 0.1)),
        momentum=tune.sample_from(lambda spec: np.random.uniform(0.85, 0.99)),
    ),
)


# Export configurations in this file
CONFIGS = dict(
    dense_mlp=DENSE_MLP,
    sparse_mlp=SPARSE_MLP,
    sparse_mlp15=SPARSE_MLP15,
    sparse_wts_mlp=SPARSE_WTS_MLP,
    sparse_act_mlp=SPARSE_ACT_MLP,
    sparsity_search=SPARSITY_SEARCH,
    sparsity_search2=SPARSITY_SEARCH2,
    sparsity_search3=SPARSITY_SEARCH3,
    sparsity_search4=SPARSITY_SEARCH4,
    sparsity_search5=SPARSITY_SEARCH5,
    sparsity_search6=SPARSITY_SEARCH6,
)
