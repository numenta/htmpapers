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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nupic.research.frameworks.dendrites.routing import (
    RoutingDataset,
    RoutingFunction,
    evaluate_dendrite_model,
    generate_context_vectors,
    train_dendrite_model,
)


class SingleLayerLinearNetwork(torch.nn.Module):
    """
    A feed-forward linear network with just a single layer and no non-linear activation
    function
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = torch.nn.Linear(input_size, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.layer(x)


def test_regular_network(
    dim_in,
    dim_out,
    num_contexts,
    dim_context,
    batch_size=64,
    num_training_epochs=1000
):
    """
    Trains and evalutes a feedforward network with no hidden layers to match an
    arbitrary routing function

    :param dim_in: the number of dimensions in the input to the routing function and
                   test module
    :param dim_out: the number of dimensions in the sparse linear output of the routing
                    function and test network
    :param num_contexts: the number of unique random binary vectors in the routing
                         function that can "route" the sparse linear output, and also
                         the number of unique context vectors
    :param dim_context: the number of dimensions in the context vectors
    :param dendrite_module: a torch.nn.Module subclass that implements a dendrite
                            module in addition to a linear feed-forward module
    :param batch_size: the batch size during training and evaluation
    :param num_training_epochs: the number of epochs for which to train the dendritic
                                network
    """

    # Input size to the model is 2 * dim_in since the context is concatenated with the
    # regular input
    model = SingleLayerLinearNetwork(input_size=2 * dim_in, output_size=dim_out)

    # Initialize routing function that this task will try to learn, and set
    # `requires_grad=False` since the routing function is static
    r = RoutingFunction(
        dim_in=dim_in,
        dim_out=dim_out,
        k=num_contexts,
        device=model.device,
        sparsity=0.7
    )
    r.sparse_weights.module.weight.requires_grad = False

    # Initialize context vectors, where each context vector corresponds to an output
    # mask in the routing function
    context_vectors = generate_context_vectors(
        num_contexts=num_contexts,
        n_dim=dim_context,
        percent_on=0.2
    )

    # Initialize datasets and dataloaders
    train_dataset = RoutingDataset(
        routing_function=r,
        input_size=r.sparse_weights.module.in_features,
        context_vectors=context_vectors,
        device=model.device,
        concat=True,
        x_min=-2.0,
        x_max=2.0
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size
    )

    test_dataset = RoutingDataset(
        routing_function=r,
        input_size=r.sparse_weights.module.in_features,
        context_vectors=context_vectors,
        device=model.device,
        concat=True,
        x_min=2.0,
        x_max=6.0
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size
    )

    # Place objects that inherit from torch.nn.Module on device
    model = model.to(model.device)
    r = r.to(r.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("epoch,mean_loss,mean_abs_err")
    for epoch in range(1, num_training_epochs + 1):

        train_dendrite_model(
            model=model,
            loader=train_dataloader,
            optimizer=optimizer,
            device=model.device,
            criterion=F.l1_loss,
            concat=True
        )

        # Validate model - note that we use a different dataset/dataloader as the input
        # distribution has changed
        results = evaluate_dendrite_model(
            model=model,
            loader=test_dataloader,
            device=model.device,
            criterion=F.l1_loss,
            concat=True
        )

        print("{},{},{}".format(
            epoch, results["loss"], results["mean_abs_err"]
        ))


if __name__ == "__main__":

    test_regular_network(
        dim_in=100,
        dim_out=100,
        num_contexts=10,
        dim_context=100
    )
