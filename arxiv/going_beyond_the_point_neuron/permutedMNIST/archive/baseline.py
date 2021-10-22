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

""" Dense MLP, sparse MLP, and dendritic MLPs on regular MNIST """

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.routing import generate_context_vectors
from nupic.research.frameworks.pytorch.models import StandardMLP
from nupic.research.frameworks.pytorch.models.common_models import SparseMLP
from nupic.research.frameworks.vernon import SupervisedExperiment as SupExp
from nupic.research.frameworks.vernon import mixins


# ------ Experiment class
class SparseSupExp(mixins.RezeroWeights, SupExp):
    """ Experiment class for sparse MLPs and dendritic MLPs """

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Generate a random context vector
        if model_type == "dendriticMLP":
            dim_context = config.get("model_args").get("dim_context")
            self.context = generate_context_vectors(num_contexts=1, n_dim=dim_context,
                                                    percent_on=0.05)
            self.context = self.context.to(self.device)


# ------ Training & evaluation functions
def train_model(exp):
    exp.model.train()
    for data, target in exp.train_loader:
        data = data.flatten(start_dim=1)

        data = data.to(exp.device)
        target = target.to(exp.device)

        # Package all items passed to model's `forward` function into a dict; this
        # includes data and context
        forward_args = [data]
        if model_type == "dendriticMLP":
            forward_args.append(
                torch.repeat_interleave(exp.context, data.size(0), dim=0)
            )

        exp.optimizer.zero_grad()
        output = exp.model(*forward_args)

        output = F.log_softmax(output)
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

        for data, target in exp.val_loader:
            data = data.flatten(start_dim=1)

            data = data.to(exp.device)
            target = target.to(exp.device)

            # Package all items passed to model's `forward` function into a dict; this
            # includes data and context
            forward_args = [data]
            if model_type == "dendriticMLP":
                forward_args.append(
                    torch.repeat_interleave(exp.context, data.size(0), dim=0)
                )

            output = exp.model(*forward_args)

            # All output units are used to compute loss / accuracy
            loss += exp.error_loss(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    mean_acc = torch.true_divide(correct, total).item() if total > 0 else 0,
    return mean_acc


def run_experiment(config):
    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)

    # ---------------------------- TRAINING & VALIDATION ---------------------------- #

    print("")
    for _epoch_id in range(1, exp.epochs + 1):
        train_model(exp)

        # Report validation accuracy
        accuracy = evaluate_model(exp)
        if isinstance(accuracy, tuple):
            accuracy = accuracy[0]

        print(f"Epoch {_epoch_id} accuracy: {accuracy}")

    print("")

    # ------------------------------------------------------------------------------- #


if __name__ == "__main__":

    # `model_type` must be one of "denseMLP", "sparseMLP", or "dendriticMLP"
    model_type = "dendriticMLP"
    assert model_type in ("denseMLP", "sparseMLP", "dendriticMLP"), "invalid model"

    model_class = dict(
        denseMLP=StandardMLP,
        sparseMLP=SparseMLP,
        dendriticMLP=DendriticMLP
    )
    model_args = dict(
        denseMLP=dict(input_size=784, num_classes=10, hidden_sizes=(2048, 2048)),
        sparseMLP=dict(input_size=784, output_size=10, kw_percent_on=(0.05, 0.05),
                       weight_sparsity=(0.05, 0.05), boost_strength=0.0,
                       boost_strength_factor=0.0, k_inference_factor=1.0,
                       use_batch_norm=False, hidden_sizes=(2048, 2048)),
        dendriticMLP=dict(input_size=784, output_size=784, hidden_sizes=[2048, 2048],
                          num_segments=1, dim_context=1024, kw=True,
                          dendrite_weight_sparsity=0.95, freeze_dendrites=False)
    )

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13062755,), (0.30810780,)),
    ])

    config = dict(
        experiment_class=SupExp if model_type == "denseMLP" else SparseSupExp,

        dataset_class=datasets.MNIST,
        dataset_args=dict(
            root=".",
            download=False,
            transform=mnist_transform,
        ),

        model_class=model_class[model_type],
        model_args=model_args[model_type],

        batch_size=256,
        val_batch_size=512,
        epochs=20,
        num_classes=10,
        distributed=False,
        seed=np.random.randint(0, 10000),

        loss_function=F.nll_loss,
        optimizer_class=torch.optim.Adam,
        optimizer_args=dict(lr=1e-3),
    )

    run_experiment(config)
