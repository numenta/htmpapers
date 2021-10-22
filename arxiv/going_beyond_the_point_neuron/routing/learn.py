# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
Perform the routing task with a dendrite layer by either (a) learning just the
dendrite weights, (b) learning both the feed-forward and dendrite weights together,
or (c) learning the feed-forward and dendrite weights, and context generation
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.research.frameworks.dendrites.routing import (
    evaluate_dendrite_model,
    init_dataloader,
    init_optimizer,
    init_test_scenario,
    train_dendrite_model,
)


def learn_to_route(
    mode,
    dim_in,
    dim_out,
    num_contexts,
    dim_context,
    dendrite_module,
    batch_size=64,
    num_training_epochs=5000,
    sparse_context_model=True,
    onehot=False,
    plot=False,
    save_interval=100,
    save_path="./models/"
):
    """
    Trains a dendrite layer to match an arbitrary routing function

    :param mode: must be one of ("dendrites", "all", "learn_context)
                 "dendrites" -> learn only dendrite weights while setting feed-forward
                 weights to those of the routing function
                 "all" -> learn both feed-forward and dendrite weights
                 "learn_context" -> learn feed-forward, dendrite, and context-generation
                 weights
    :param dim_in: the number of dimensions in the input to the routing function and
                   test module
    :param dim_out: the number of dimensions in the sparse linear output of the routing
                    function and test layer
    :param num_contexts: the number of unique random binary vectors in the routing
                         function that can "route" the sparse linear output, and also
                         the number of unique context vectors
    :param dim_context: the number of dimensions in the context vectors
    :param dendrite_module: a torch.nn.Module subclass that implements a dendrite
                            module in addition to a linear feed-forward module
    :param batch_size: the batch size during training and evaluation
    :param num_training_epochs: the number of epochs for which to train the dendrite
                                layer
    :param sparse_context_model: whether to use a sparse MLP to generate the context;
                                 applicable if mode == "learn_context"
    :param onehot: whether the context integer should be encoded as a onehot vector
                   when input into the context generation model
    :param plot: whether to plot a loss curve
    :param save_interval: number of epochs between saving the model
    :param save_path: path to folder in which to save model checkpoints.
    """

    r, dendrite_layer, context_model, context_vectors, device = init_test_scenario(
        mode=mode,
        dim_in=dim_in,
        dim_out=dim_out,
        num_contexts=num_contexts,
        dim_context=dim_context,
        dendrite_module=dendrite_module,
        sparse_context_model=sparse_context_model,
        onehot=onehot
    )

    train_dataloader = init_dataloader(
        routing_function=r,
        context_vectors=context_vectors,
        device=device,
        batch_size=batch_size,
        x_min=-2.0,
        x_max=2.0,
    )

    test_dataloader = init_dataloader(
        routing_function=r,
        context_vectors=context_vectors,
        device=device,
        batch_size=batch_size,
        x_min=2.0,
        x_max=6.0,
    )

    optimizer = init_optimizer(mode=mode,
                               layer=dendrite_layer,
                               context_model=context_model)

    print("epoch,mean_loss,mean_abs_err")
    losses = []

    for epoch in range(1, num_training_epochs + 1):

        l1_weight_decay = None
        # Select L1 weight decay penalty based on scenario
        if mode == "dendrites":
            l1_weight_decay = 0.0
        elif mode == "all" or mode == "learn_context":
            l1_weight_decay = 1e-6

        train_dendrite_model(
            model=dendrite_layer,
            context_model=context_model,
            loader=train_dataloader,
            optimizer=optimizer,
            device=device,
            criterion=F.l1_loss,
            concat=False,
            l1_weight_decay=l1_weight_decay,
        )

        # Validate model
        results = evaluate_dendrite_model(
            model=dendrite_layer,
            context_model=context_model,
            loader=test_dataloader,
            device=device,
            criterion=F.l1_loss,
            concat=False
        )

        print("{},{}".format(
            epoch, results["mean_abs_err"]
        ))
        # track loss for plotting
        if plot:
            losses.append(results["mean_abs_err"])
        # save models for future visualization or training
        if save_interval and epoch % save_interval == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(dendrite_layer.state_dict(),
                       save_path + "dendrite_" + str(epoch))
            if context_model:
                torch.save(context_model.state_dict(),
                           save_path + "context_" + str(epoch))

    if plot:
        losses = np.array(losses)
        plt.scatter(x=np.arange(1, num_training_epochs + 1), y=losses)
        plt.savefig("training_curve.png")


if __name__ == "__main__":

    # Learn dendrite weights that learn to route, while keeping feedforward weights
    # fixed

    mode = "all"
    # whether input to context model should be a onehot vector
    onehot = True
    # whether context model should be a sparse MLP
    sparse_context_model = True

    learn_to_route(
        mode=mode,
        dim_in=100,
        dim_out=100,
        num_contexts=10,
        dim_context=100,
        dendrite_module=AbsoluteMaxGatingDendriticLayer,
        num_training_epochs=5000,
        plot=True,
        save_interval=100,
        onehot=onehot,
        sparse_context_model=sparse_context_model
    )
