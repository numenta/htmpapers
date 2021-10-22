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
import torch.nn as nn

# from garage.torch import NonLinearity
from nupic.research.frameworks.pytorch.models.le_sparse_net import (
    add_sparse_linear_layer,
)


class MultiHeadedSparseMLP(nn.Module):
    def __init__(self, input_size,
                 num_heads,
                 output_dims,
                 output_nonlinearities=(None, None),
                 hidden_sizes=(32, 32),
                 linear_activity_percent_on=(0.1, 0.1),
                 linear_weight_percent_on=(0.4, 0.4),
                 boost_strength=1.67,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 consolidated_sparse_weights=False,
                 ):
        super(MultiHeadedSparseMLP, self).__init__()
        assert len(output_dims) == len(output_nonlinearities) == num_heads
        assert len(hidden_sizes) == len(linear_weight_percent_on)
        assert len(linear_weight_percent_on) == len(linear_weight_percent_on)
        self.num_heads = num_heads

        self._hidden_base = nn.Sequential()
        self._hidden_base.add_module("flatten", nn.Flatten())
        # Add Sparse Linear layers
        for i in range(len(hidden_sizes)):
            add_sparse_linear_layer(
                network=self._hidden_base,
                suffix=i + 1,
                input_size=input_size,
                linear_n=hidden_sizes[i],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                weight_sparsity=linear_weight_percent_on[i],
                percent_on=linear_activity_percent_on[i],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
                consolidated_sparse_weights=consolidated_sparse_weights,
            )
            input_size = hidden_sizes[i]

        self._output_layers = nn.ModuleList()
        for i in range(self.num_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(input_size, output_dims[i])
            output_layer.add_module("linear", linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module("non_linearity",
                                        None)
            self._output_layers.append(output_layer)

    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = self._hidden_base(input_val)

        return [output_layer(x) for output_layer in self._output_layers]
