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
This module trains & evaluates a dendritic network in a continual learning setting on
permutedMNIST for a specified number of tasks/permutations. A context vector is
provided to the dendritic network, so task information need not be inferred.

This setup is very similar to that of context-dependent gating model from the paper
'Alleviating catastrophic forgetting using contextdependent gating and synaptic
stabilization' (Masse et al., 2018).
"""

import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.pytorch.datasets import ContextDependentPermutedMNIST
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins


# ------ Experiment class
class PermutedMNISTExperiment(mixins.RezeroWeights,
                              ContinualLearningExperiment):
    pass


# ------ Training & evaluation functions
def train_model(exp):
    exp.model.train()
    for (data, context), target in exp.train_loader:
        data = data.flatten(start_dim=1)

        # Since there's only one output head, target values should be modified to be in
        # the range [0, 1, ..., 9]
        target = target % exp.num_classes_per_task

        data = data.to(exp.device)
        context = context.to(exp.device)
        target = target.to(exp.device)

        exp.optimizer.zero_grad()
        output = exp.model(data, context)

        error_loss = exp.error_loss(output, target)
        error_loss.backward()
        exp.optimizer.step()

        # Rezero weights if necessary
        exp.post_optimizer_step(exp.model)


def evaluate_model(exp):
    exp.model.eval()
    total = 0

    loss = torch.tensor(0., device=exp.device)
    correct = torch.tensor(0, device=exp.device)

    with torch.no_grad():

        for (data, context), target in exp.val_loader:
            data = data.flatten(start_dim=1)

            # Since there's only one output head, target values should be modified to
            # be in the range [0, 1, ..., 9]
            target = target % exp.num_classes_per_task

            data = data.to(exp.device)
            context = context.to(exp.device)
            target = target.to(exp.device)

            output = exp.model(data, context)

            # All output units are used to compute loss / accuracy
            loss += exp.error_loss(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    mean_acc = torch.true_divide(correct, total).item() if total > 0 else 0
    return mean_acc


def run_experiment(config):
    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)

    exp.model = exp.model.to(exp.device)

    # Read optimizer class and args from config as it will be used to reinitialize the
    # model's optimizer
    optimizer_class = config.get("optimizer_class", torch.optim.SGD)
    optimizer_args = config.get("optimizer_args", {})

    # --------------------------- CONTINUAL LEARNING PHASE -------------------------- #

    for task_id in range(num_tasks):

        # Train model on current task
        exp.train_loader.sampler.set_active_tasks(task_id)
        for _epoch_id in range(exp.epochs):
            train_model(exp)

        if task_id in config["epochs_to_validate"]:

            print("")
            print(f"=== AFTER TASK {task_id} ===")
            print("")

            # Evaluate model accuracy on each task separately
            for eval_task_id in range(task_id + 1):

                exp.val_loader.sampler.set_active_tasks(eval_task_id)
                acc_task = evaluate_model(exp)
                if isinstance(acc_task, tuple):
                    acc_task = acc_task[0]

                print(f"task {eval_task_id} accuracy: {acc_task}")
            print("")

        else:
            print(f"--Completed task {task_id}--")

        # Reset optimizer before starting new task
        del exp.optimizer
        exp.optimizer = optimizer_class(exp.model.parameters(), **optimizer_args)

    # ------------------------------------------------------------------------------- #

    # Report final aggregate accuracy
    exp.val_loader.sampler.set_active_tasks(range(num_tasks))
    acc_task = evaluate_model(exp)
    if isinstance(acc_task, tuple):
        acc_task = acc_task[0]

    print(f"Final test accuracy: {acc_task}")
    print("")


if __name__ == "__main__":

    num_tasks = 2

    config = dict(
        experiment_class=PermutedMNISTExperiment,

        dataset_class=ContextDependentPermutedMNIST,
        dataset_args=dict(
            num_tasks=num_tasks,
            dim_context=1024,
            seed=42,
            download=True,
        ),

        model_class=DendriticMLP,
        model_args=dict(
            input_size=784,
            output_size=10,
            hidden_sizes=[64, 64],
            num_segments=num_tasks,
            dim_context=1024,  # Note: with the Gaussian dataset, `dim_context` was
                               # 2048, but this shouldn't effect results
            kw=True,
            # dendrite_sparsity=0.0,
        ),

        batch_size=256,
        val_batch_size=512,
        epochs=1,
        epochs_to_validate=(4, 9, 24, 49),  # Note: `epochs_to_validate` is treated as
                                            # the set of task ids after which to
                                            # evaluate the model on all seen tasks
        num_tasks=num_tasks,
        num_classes=10 * num_tasks,
        distributed=False,
        seed=42,

        loss_function=F.cross_entropy,
        optimizer_class=torch.optim.Adam,
        optimizer_args=dict(lr=0.001),
    )

    run_experiment(config)
