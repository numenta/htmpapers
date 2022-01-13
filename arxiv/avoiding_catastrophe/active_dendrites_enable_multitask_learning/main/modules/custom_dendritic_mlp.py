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

import torch
from torch import nn

from main.dendrites import ModularDendriticMLP

from nupic.torch.modules import KWinners, SparseWeights


class CustomDendriticMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 context_dim,
                 output_sizes,
                 hidden_sizes,
                 layers_modulated,
                 num_segments,
                 kw_percent_on,
                 context_percent_on,
                 weight_sparsity,
                 weight_init,
                 dendrite_weight_sparsity,
                 dendrite_init,
                 dendritic_layer_class,
                 output_nonlinearity,
                 preprocess_module_type,
                 preprocess_output_dim,
                 preprocess_kw_percent_on
                 ):
        super().__init__()

        self.context_dim = context_dim
        self.weight_sparsity = weight_sparsity
        self.preprocess_module_type = preprocess_module_type

        # preprocess module: builds a representation of context (as input to dendrite segments)
        self.preprocess_module, preprocess_output_dim = self._create_preprocess_module(
            preprocess_module_type,
            preprocess_output_dim,
            preprocess_kw_percent_on
        )

        self.dendritic_module = ModularDendriticMLP(
            input_size=input_dim,
            context_size=preprocess_output_dim,
            output_size=output_sizes,
            hidden_sizes=hidden_sizes,
            layers_modulated=layers_modulated,
            num_segments=num_segments,
            kw_percent_on=kw_percent_on,
            context_percent_on=context_percent_on,
            weight_sparsity=weight_sparsity,
            weight_init=weight_init,
            dendrite_weight_sparsity=dendrite_weight_sparsity,
            dendrite_init=dendrite_init,
            dendritic_layer_class=dendritic_layer_class,
            output_nonlinearity=output_nonlinearity,
        )

    def _create_preprocess_module(self, module_type, preprocess_output_dim, kw_percent_on):
        preprocess_module = nn.Sequential()

        if module_type is None:
            return preprocess_module, self.context_dim

        linear_layer = torch.nn.Linear(
            self.context_dim,
            preprocess_output_dim,
            bias=True
        )

        if module_type == "relu":
            nonlinearity = nn.ReLU()
        elif module_type == "kw":
            nonlinearity = KWinners(
                n=preprocess_output_dim,
                percent_on=kw_percent_on,
                k_inference_factor=1.0,
                boost_strength=0.0,
                boost_strength_factor=0.0
            )
        else:
            nonlinearity = nn.Identity()

        preprocess_module.add_module("linear_layer", linear_layer)
        preprocess_module.add_module("nonlinearity", nonlinearity)

        return preprocess_module, preprocess_output_dim

    def forward(self, x, context):
        return self.dendritic_module(x, self.preprocess_module(context))