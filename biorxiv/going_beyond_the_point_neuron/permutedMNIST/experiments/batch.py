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
These are baseline experiments testing how well the sparse network without any dendrites
performs on a batch version of permuted MNIST. The goal is to see if the base
feed-forward network has sufficient capacity to learn all num_tasks*60000 training
examples simultaneously.

This accuracy could be a reasonable upper bound for the continual learning version,
which has to have learned all the same examples by the end of the experiment.
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


class NoDendriteBatchExperiment(mixins.RezeroWeights,
                                mixins.PermutedMNISTTaskIndices,
                                DendriteContinualLearningExperiment):

    def should_stop(self):
        return self.current_epoch >= self.epochs

    def run_task(self):
        self.train_loader.sampler.set_active_tasks(range(self.num_tasks))
        self.val_loader.sampler.set_active_tasks(range(self.num_tasks))
        self.logger.info("Training task %d, epoch %d...",
                         self.current_task, self.current_epoch)

        # Just run one epoch
        ret = self.run_epoch()

        print("run_task ret: ", ret)
        return ret


# Run two MNIST tasks in batch mode
SPARSE_BATCH_2 = dict(
    experiment_class=NoDendriteBatchExperiment,

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
        kw_percent_on=0.1,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),

    batch_size=128,
    val_batch_size=512,
    epochs=3,                       # This is the total number of epochs run
    epochs_to_validate=range(100),
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
            name="sparse_batch_2",
            group="sparse_batch_2",
            notes="""
            Sparse network with batch permuted MNIST.
            """
        )
    ),

)

SPARSE_BATCH_50 = deepcopy(SPARSE_BATCH_2)
SPARSE_BATCH_50["dataset_args"].update(num_tasks=50)
SPARSE_BATCH_50["env_config"]["wandb"].update(
    name="sparse_batch_50_3x",
    group="sparse_batch_50",
)
SPARSE_BATCH_50.update(
    batches_in_epoch=1404,  # About 3*60K samples/epoch, completes in num_tasks epochs
    epochs=50,
    num_tasks=50,
    num_classes=10 * 50,
    num_samples=1,
)

# Export configurations in this file
CONFIGS = dict(
    sparse_batch_2=SPARSE_BATCH_2,
    sparse_batch_50=SPARSE_BATCH_50,
)
