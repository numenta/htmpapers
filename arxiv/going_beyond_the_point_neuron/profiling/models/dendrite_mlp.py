# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------
from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from projects.dendrites.profiling.models.multiheaded_dendrite_mlp import (
    MultiHeadedDendriticMLP,
)


class DendriticMLP(MultiHeadedDendriticMLP):
    """
    A dendritic network which is similar to a MLP with a two hidden layers, except that
    activations are modified by dendrites. The context input to the network is used as
    input to the dendritic weights.
    """

    def __init__(self,
                 input_size,
                 output_dim,
                 dim_context,
                 hidden_sizes=(32, 32),
                 num_segments=(5, 5),
                 sparsity=0.5,
                 k_winners=False,
                 k_winner_percent_on=0.1,
                 relu=False,
                 output_nonlinearities=(None, ),
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):

        super(DendriticMLP, self).__init__(
            input_size=input_size,
            num_heads=1,
            output_dims=(output_dim, ),
            dim_context=dim_context,
            hidden_sizes=hidden_sizes,
            num_segments=num_segments,
            sparsity=sparsity,
            k_winners=k_winners,
            relu=relu,
            k_winners_percent_on=k_winner_percent_on,
            output_nonlinearities=output_nonlinearities,
            dendritic_layer_class=dendritic_layer_class
        )
        self._output_dim = output_dim

    # @profile
    def forward(self, x, context):
        return super().forward(x, context)[0]

    # @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim
