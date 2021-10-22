# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
This module runs a dendritic network in continual learning setting where each task
consists of learning to classify samples drawn from one of two multivariate normal
distributions.

Dendritic weights can either be hardcoded (to induce overlapping or non-overlapping
subnetworks) or learned. All output heads are used for both training and inference.

Usage: adjust the config parameters `kw`, `dendrite_sparsity`, `weight_init`,
`dendrite_init`, and `freeze_dendrites` (all in `model_args`).
"""

import pprint
import time

import numpy as np
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites.modules import DendriticMLP
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins
from nupic.torch.duty_cycle_metrics import max_entropy
from projects.dendrites.gaussian_classification.gaussian import GaussianDataset


# ------ Experiment class
class DendritesExperiment(mixins.RezeroWeights,
                          ContinualLearningExperiment):

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Manually set dendritic weights to invoke subnetworks; if the user sets
        # `freeze_dendrites=True`, we assume dendritic weights are intended to be
        # hardcoded
        if self.model.freeze_dendrites:
            self.model.hardcode_dendritic_weights(
                context_vectors=self.train_loader.dataset._contexts, init="overlapping"
            )


# ------ Training & evaluation function
def train_model(exp):
    # Assume `loader` yields 3-item tuples of the form (data, context, target)
    exp.model.train()
    for (data, context), target in exp.train_loader:
        data = data.to(exp.device)
        context = context.to(exp.device)
        target = target.to(exp.device)

        exp.optimizer.zero_grad()
        output = exp.model(data, context)

        # Outputs are placed through a log softmax since `error_loss` is `F.nll_loss`,
        # which assumes it will receive 'logged' values
        output = F.log_softmax(output)
        error_loss = exp.error_loss(output, target)
        error_loss.backward()
        exp.optimizer.step()

        # Rezero weights if necessary
        exp.post_optimizer_step(exp.model)


def evaluate_model(exp):
    # Assume `loader` yields 3-item tuples of the form (data, context, target)
    exp.model.eval()
    total = 0

    loss = torch.tensor(0., device=exp.device)
    correct = torch.tensor(0, device=exp.device)

    with torch.no_grad():

        for (data, context), target in exp.val_loader:
            data = data.to(exp.device)
            context = context.to(exp.device)
            target = target.to(exp.device)

            output = exp.model(data, context)

            # All output units are used to compute loss / accuracy
            loss += exp.error_loss(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    mean_acc = torch.true_divide(correct, total).item() if total > 0 else 0,
    return mean_acc


def run_experiment(config):
    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)

    exp.model = exp.model.to(exp.device)

    # --------------------------- CONTINUAL LEARNING PHASE -------------------------- #
    for task_id in range(num_tasks):

        # Train model on current task
        t1 = time.time()
        exp.train_loader.sampler.set_active_tasks(task_id)
        for _epoch_id in range(num_epochs):
            train_model(exp)
        t2 = time.time()

        print(f"train time [task {task_id}]: {t2 - t1}")

        # Evaluate model accuracy on each task separately
        if task_id in config["epochs_to_validate"]:

            print(f"\n=== AFTER TASK {task_id} ===\n")

            for eval_task_id in range(task_id + 1):

                exp.val_loader.sampler.set_active_tasks(eval_task_id)
                acc_task = evaluate_model(exp)
                if isinstance(acc_task, tuple):
                    acc_task = acc_task[0]

                print(f"task {eval_task_id} accuracy: {acc_task}")

            t3 = time.time()
            print(f"\nevaluation time: {t3 - t2}")
            print(f"====================\n")

    # ------------------------------------------------------------------------------- #

    # Report final aggregate accuracy
    exp.val_loader.sampler.set_active_tasks(range(num_tasks))
    acc_task = evaluate_model(exp)
    if isinstance(acc_task, tuple):
        acc_task = acc_task[0]

    print(f"Final test accuracy: {acc_task}")

    # Print entropy of layers
    max_possible_entropy = max_entropy(exp.model.hidden_size,
                                       int(0.05 * exp.model.hidden_size))
    if exp.model.kw:
        print(f"   KW1 entropy: {exp.model.kw1.entropy().item()}")
        print(f"   KW2 entropy: {exp.model.kw2.entropy().item()}")
        print(f"   max entropy: {max_possible_entropy}")
    print("")


if __name__ == "__main__":

    num_tasks = 50
    num_epochs = 1  # Number of training epochs per task

    config = dict(
        experiment_class=DendritesExperiment,

        dataset_class=GaussianDataset,
        dataset_args=dict(
            num_classes=2 * num_tasks,
            num_tasks=num_tasks,
            training_examples_per_class=2500,
            validation_examples_per_class=500,
            dim_x=2048,
            dim_context=2048,
            seed=np.random.randint(0, 1000),
        ),

        model_class=DendriticMLP,
        model_args=dict(
            input_size=2048,
            output_size=2 * num_tasks,
            hidden_size=2048,
            num_segments=num_tasks,
            dim_context=2048,
            kw=True,  # Turning on k-Winners when hardcoding dendrites to induce
                      # non-overlapping subnetworks results in 5% winners
            dendrite_sparsity=0.0,  # Irrelevant if `freeze_dendrites=True`
            weight_init="modified",  # Must be one of {"kaiming", "modified"}
            dendrite_init="modified",  # Irrelevant if `freeze_dendrites=True`
            freeze_dendrites=False
        ),

        batch_size=64,
        val_batch_size=512,
        epochs=num_epochs,
        epochs_to_validate=[0, 3, 6, 10, 20, num_tasks - 1],
        num_tasks=num_tasks,
        num_classes=2 * num_tasks,
        distributed=False,
        seed=np.random.randint(0, 10000),

        optimizer_class=torch.optim.SGD,
        optimizer_args=dict(lr=2e-1),
    )

    print("Experiment config: ")
    pprint.pprint(config)
    print("")

    run_experiment(config)
