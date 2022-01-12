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

from collections import Iterable

import numpy as np
import torch
from torch import nn

from nupic.torch.modules import KWinners, SparseWeights, rezero_weights

from main.dendrites import FFLayer

class SequentialBlock(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input


class ModularDendriticMLP(nn.Module):
    """
    A simple but restricted MLP with two hidden layers of the same size. Each hidden
    layer contains units with dendrites. Dendrite segments receive context directly as
    input.  The class is used to experiment with different dendritic weight
    initializations and learning parameters

    :param input_size: size of the input to the network
    :param context_size: size of the context to the network
    :param output_size: the number of units in the output layer. Must be either an
                        integer if there is a single output head, or an iterable
                        of integers if there are multiple output heads.
    :param hidden_sizes: the number of units in each hidden layer
    :param layers_modulated: the layers which are modulated by dendrites
    :param num_segments: the number of dendritic segments that each hidden unit has
    :param kw_percent_on: percent of hidden units activated by K-winners. If 0, use ReLU
    :param context_percent_on: percent of non-zero units in the context input.
    :param weight_sparsity: the sparsity level of feed-forward weights.
    :param weight_init: the initialization applied to feed-forward weights; must be
                        either "kaiming" (for Kaiming Uniform) of "modified" (for
                        sparse Kaiming Uniform)
    :param dendrite_weight_sparsity: the sparsity level of dendritic weights.
    :param dendrite_init: the initialization applied to dendritic weights; similar to
                          `weight_init`
    :param dendritic_layer_class: dendritic layer class to use for each hidden layer
    :param output_nonlinearity: nonlinearity to apply to final output layer.
                                'None' of no nonlinearity.
    :param freeze_dendrites: whether to set `requires_grad=False` for all dendritic
                             weights so they don't train

                    _____
                   |_____|    # classifier layer, no dendrite input
                      ^
                      |
                  _________
    context -->  |_________|  # second linear layer with dendrites
                      ^
                      |
                  _________
    context -->  |_________|  # first linear layer with dendrites
                      ^
                      |
                    input
    """

    def __init__(
        self,
        input_size,
        context_size,
        output_size,
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
        freeze_dendrites=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.context_size = context_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.layers_modulated = layers_modulated
        self.num_segments = num_segments
        self.kw_percent_on = kw_percent_on
        self.context_percent_on = context_percent_on
        self.weight_sparsity = weight_sparsity
        self.weight_init = weight_init
        self.dendrite_weight_sparsity = dendrite_weight_sparsity
        self.dendrite_init = dendrite_init
        self.output_nonlinearity = output_nonlinearity

        self.layers = nn.ModuleList()

        for i in range(len(self.hidden_sizes)):
            block_name = ""

            if i not in self.layers_modulated:
                linear = FFLayer(
                    module=nn.Linear(input_size, self.hidden_sizes[i], bias=True),
                    module_sparsity=self.weight_sparsity,
                )
                block_name = "ff"
            else:
                linear = dendritic_layer_class(
                    module=nn.Linear(input_size, self.hidden_sizes[i], bias=True),
                    num_segments=self.num_segments,
                    dim_context=self.context_size,
                    module_sparsity=self.weight_sparsity,
                    dendrite_sparsity=self.dendrite_weight_sparsity,
                )
                block_name = "dendrite"

                if self.dendrite_init == "modified":
                    self._init_sparse_dendrites(linear, 1 - self.context_percent_on)

                if freeze_dendrites:
                    # Dendritic weights will not be updated during backward pass
                    for name, param in linear.named_parameters():
                        if "segments" in name:
                            param.requires_grad = False

            if self.weight_init == "modified":
                # Scale weights to be sampled from the new initialization U(-h, h) where
                # h = sqrt(1 / (weight_density * previous_layer_percent_on))

                # first hidden layer can't have kw input
                if i == 0:
                    self._init_sparse_weights(linear, 0.0)
                else:
                    self._init_sparse_weights(
                        linear,
                        1 - self.kw_percent_on if self.kw_percent_on else 0.0
                    )

            if self.kw_percent_on:
                activation = KWinners(n=hidden_sizes[i],
                                      percent_on=kw_percent_on,
                                      k_inference_factor=1.0,
                                      boost_strength=0.0,
                                      boost_strength_factor=0.0)
            else:
                activation = nn.ReLU()

            block = SequentialBlock()
            block.add_module(block_name, SequentialBlock(linear, activation))
            self.layers.append(block)

            input_size = self.hidden_sizes[i]


        if not isinstance(output_size, Iterable):
            output_size = (output_size,)

        self._output_layers = nn.ModuleList()

        for out_size in output_size:
            output_layer = nn.Sequential()

            output_linear = SparseWeights(module=nn.Linear(input_size, out_size),
                                          sparsity=self.weight_sparsity, allow_extremes=True)

            if self.weight_init == "modified":
                self._init_sparse_weights(
                    output_linear,
                    1 - self.kw_percent_on if self.kw_percent_on else 0.0
                )

            output_layer.add_module("output_linear", output_linear)

            if self.output_nonlinearity is not None:
                output_layer.add_module("non_linearity", self.output_nonlinearity)

            self._output_layers.append(output_layer)

    def forward(self, x, context):
        for layer in self.layers:
            x = layer(x, context)

        if len(self._output_layers) == 1:
            return self._output_layers[0](x)
        else:
            return [out_layer(x) for out_layer in self._output_layers]


    # ------ Weight initialization functions ------
    @staticmethod
    def _init_sparse_weights(m, input_sparsity):
        """
        Modified Kaiming weight initialization that considers input sparsity and weight
        sparsity.
        """
        input_density = 1.0 - input_sparsity
        weight_density = 1.0 - m.sparsity
        _, fan_in = m.module.weight.size()
        bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
        nn.init.uniform_(m.module.weight, -bound, bound)
        m.apply(rezero_weights)

    @staticmethod
    def _init_sparse_dendrites(m, input_sparsity):
        """
        Modified Kaiming initialization for dendrites segments that consider input
        sparsity and dendritic weight sparsity.
        """
        # Assume `m` is an instance of `DendriticLayerBase`
        if m.segments is not None:
            input_density = 1.0 - input_sparsity
            weight_density = 1.0 - m.segments.sparsity
            fan_in = m.dim_context
            bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
            nn.init.uniform_(m.segment_weights, -bound, bound)
            m.apply(rezero_weights)

    def hardcode_dendritic_weights(self, context_vectors, init):
        """
        Set up specific weights for each dendritic segment based on the value of init.

        if init == "overlapping":
            We hardcode the weights of dendrites such that each context selects 5% of
            hidden units to become active and form a subnetwork. Hidden units are
            sampled with replacement, hence subnetworks can overlap. Any context/task
            which does not use a particular hidden unit will cause it to turn off, as
            the unit's other segment(s) have -1 in all entries and will yield an
            extremely small dendritic activation.

        otherwise if init == "non_overlapping":
            We hardcode the weights of dendrites such that each unit recognizes a single
            random context vector. The first dendritic segment is initialized to contain
            positive weights from that context vector. The other segment(s) ensure that
            the unit is turned off for any other context - they contain negative weights
            for all other weights.

        :param context_vectors:
        :param init: a string "overlapping" or "non_overlapping"
        """
        if self.num_segments > 0:
            for dendrite in self._layers:
                self._hardcode_dendritic_weights(dendrite.weights, context_vectors,
                                                 init)

    @staticmethod
    def _hardcode_dendritic_weights(dendrite_weights, context_vectors, init):
        squeeze = False
        if len(dendrite_weights.shape) == 2:
            # 1 segment dendrite, so add in a segment dimension
            squeeze = True
            original_weights = dendrite_weights
            dendrite_weights = dendrite_weights.unsqueeze(dim=1)

        num_units, num_segments, dim_context = dendrite_weights.size()
        num_contexts, _ = context_vectors.size()

        if init == "overlapping":
            new_dendritic_weights = -0.95 * torch.ones((num_units, num_segments,
                                                        dim_context))

            # The number of units to allocate to each context (with replacement)
            k = int(0.05 * num_units)

            # Keep track of the number of contexts for which each segment has already
            # been chosen; this is to not overwrite a previously hardcoded segment
            num_contexts_chosen = {i: 0 for i in range(num_units)}

            for c in range(num_contexts):

                # Pick k random units to be activated by the cth context
                selected_units = torch.randperm(num_units)[:k]
                for i in selected_units:
                    i = i.item()

                    # If num_segments other contexts have already selected unit i to
                    # become active, skip
                    segment_id = num_contexts_chosen[i]
                    if segment_id == num_segments:
                        continue

                    new_dendritic_weights[i, segment_id, :] = context_vectors[c, :]
                    num_contexts_chosen[i] += 1

        elif init == "non_overlapping":
            new_dendritic_weights = torch.zeros((num_units, num_segments, dim_context))

            for i in range(num_units):
                context_perm = context_vectors[torch.randperm(num_contexts), :]
                new_dendritic_weights[i, :, :] = 1.0 * (context_perm[0, :] > 0)
                new_dendritic_weights[i, 1:, :] = -1
                new_dendritic_weights[i, 1:, :] += new_dendritic_weights[i, 0, :]
                del context_perm

        else:
            raise Exception("Invalid dendritic weight hardcode choice")

        dendrite_weights.data = new_dendritic_weights

        if squeeze:
            dendrite_weights = dendrite_weights.squeeze(dim=1)
            # dendrite weights doesn't point to the dendrite weights tensor,
            # so expicitly assign the new values
            original_weights.data = dendrite_weights