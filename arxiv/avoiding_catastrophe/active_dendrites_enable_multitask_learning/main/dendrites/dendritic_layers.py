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
A simple implementation of dendrite weights. This combines the output from a (sparse)
linear layer with the output from a set of dendritic segments.
"""
import abc
import math
import itertools
from collections.abc import Iterable

import numpy as np
import torch
from nupic.torch.modules.sparse_weights import HasRezeroWeights, SparseWeights

from .dendritic_functions import (dendritic_absolute_max_gate_1d,
                                  dendritic_bias_1d)


class DendriticBias1d(torch.nn.Module):
    def forward(self, y, dendrite_activations):
        return dendritic_bias_1d(y, dendrite_activations)


class DendriticAbsoluteMaxGate1d(torch.nn.Module):
    def forward(self, y, dendrite_activations):
        return dendritic_absolute_max_gate_1d(y, dendrite_activations)

class DendriteSegments(torch.nn.Module, HasRezeroWeights):
    """
    This implements dendrite segments over a set of units. Each unit has a set of
    segments modeled by a linear transformation from a context vector to output value
    for each segment.

    Ported from nupic.research @ https://github.com/numenta/nupic.research
    """

    def __init__(self, num_units, num_segments, dim_context, sparsity, bias=None):
        """
        :param num_units: number of units i.e. neurons;
                          each unit will have it's own set of dendrite segments
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param num_segments: number of dendrite segments per unit
        :param sparsity: sparsity of connections;
                        this is over each linear transformation from
                        dim_context to num_segments
        """
        super().__init__()

        # Save params.
        self.num_units = num_units
        self.num_segments = num_segments
        self.dim_context = dim_context
        self.sparsity = sparsity

        # TODO: Use named dimensions.
        weights = torch.Tensor(num_units, num_segments, dim_context)
        self.weights = torch.nn.Parameter(weights)

        # Create a bias per unit per segment.
        if bias:
            biases = torch.Tensor(num_units, num_segments)
            self.biases = torch.nn.Parameter(biases)
        else:
            self.register_parameter("biases", None)
        self.reset_parameters()

        # Create a random mask per unit per segment (dims=[0, 1])
        zero_mask = random_mask(
            self.weights.shape,
            sparsity=sparsity,
            dims=[0, 1]
        )

        # Use float16 because pytorch distributed nccl doesn't support bools.
        self.register_buffer("zero_mask", zero_mask.half())

        self.rezero_weights()

    def extra_repr(self):
        return (
            f"num_units={self.num_units}, "
            f"num_segments={self.num_segments}, "
            f"dim_context={self.dim_context}, "
            f"sparsity={self.sparsity}, "
            f"bias={self.biases is not None}"
        )

    def reset_parameters(self):
        """Initialize the linear transformation for each unit."""
        for unit in range(self.num_units):
            weight = self.weights[unit, ...]
            if self.biases is not None:
                bias = self.biases[unit, ...]
            else:
                bias = None
            init_linear_(weight, bias)

    def rezero_weights(self):
        self.weights.data.masked_fill_(self.zero_mask.bool(), 0)

    def forward(self, context):
        """
        Matrix-multiply the context with the weight tensor for each dendrite segment.
        This is done for each unit and so the output is of length num_units.
        """

        # Matrix multiply using einsum:
        #    * b => the batch dimension
        #    * k => the context dimension; multiplication will be along this dimension
        #    * ij => the units and segment dimensions, respectively
        # W^C * M^C * C -> num_units x num_segments
        output = torch.einsum("ijk,bk->bij", self.weights, context)

        if self.biases is not None:
            output += self.biases
        return output


def init_linear_(weight, bias=None):
    """
    Performs the default initilization of a weight and bias parameter
    of a linear layaer; done in-place.
    """
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)


def random_mask(size, sparsity, dims=None, **kwargs):
    """
    This creates a random off-mask (True => off) of 'size' with the specified 'sparsity'
    level along 'dims'. If 'dims' is 1, for instance, then `mask[:, d, ...]` has the
    desired sparsity for all d. If dims is a list, say [0, 1], then `mask[d1, d2, ...]`
    will have the desired sparsity level for all d1 and d2. If None, the sparsity is
    applied over the whole tensor.
    :param size: shape of tensor
    :param sparsity: fraction of non-zeros
    :param dims: which dimensions to apply the sparsity
    :type dims: int or iterable
    :param kwargs: keywords args passed to torch.ones;
                   helpful for specifying device, for instace
    """

    assert 0 <= sparsity <= 1

    # Start with all elements off.
    mask = torch.ones(size, **kwargs)

    # Find sparse submasks along dims; recursively call 'random_mask'.
    if dims is not None:
        if not isinstance(dims, Iterable):
            dims = [dims]

        # Loop all combinations that index through dims.
        # The 1D case is equivalent to range.
        dim_lengths = [mask.shape[dim] for dim in dims]
        dim_indices = itertools.product(*[range(dl) for dl in dim_lengths])

        for idxs in dim_indices:

            # For example, this may yield a slice that gives
            # `mask[dim_slice] == mask[:, 0, 0]` where `dims=[1, 2]`.
            dim_slice = [
                idxs[dims.index(d)] if d in dims else slice(None)
                for d in range(len(mask.shape))
            ]

            # Assign the desired sparsity to the submask.
            sub_mask = mask[dim_slice]
            sub_mask[:] = random_mask(
                sub_mask.shape,
                sparsity, **kwargs, dims=None
            )

        return mask

    # Randomly choose indices to make non-zero ("nz").
    mask_flat = mask.view(-1)  # flattened view
    num_total = mask_flat.shape[0]
    num_nz = int(round((1 - sparsity) * num_total))
    on_indices = np.random.choice(num_total, num_nz, replace=False)
    mask_flat[on_indices] = False

    return mask


class DendriticLayerBase(SparseWeights, metaclass=abc.ABCMeta):
    """
    Base class for all Dendritic Layer modules.
    This combines a DendriteSegments module with a SparseLinear module.
    The output from the dendrite segments (shape of num_units x num_segments)
    is applied to the output of of the linear weights (shape of num_units).
    Thus, each linear output unit gets modulated by a set of dendritic segments.

    Ported from nupic.research @ https://github.com/numenta/nupic.research
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        TODO: specify the type - what is module_sparsity type?
        :param module: linear module from in-units to out-units
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param dendrite_bias: whether or not dendrite activations have an additive bias
        """
        self.dim_context = dim_context
        self.segments = None
        super().__init__(
            module,
            sparsity=module_sparsity,
            allow_extremes=True
        )

        self.segments = DendriteSegments(
            num_units=module.weight.shape[0],
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    @abc.abstractmethod
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites using function specified by subclass"""
        raise NotImplementedError

    def forward(self, x, context):
        """Compute of linear layer and apply output of dendrite segments."""
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)

    @property
    def segment_weights(self):
        return self.segments.weights


class OneSegmentDendriticLayer(SparseWeights):
    """
    Class for a layer of units with exactly one sparse dendritic segment per unit. With
    this assumption the segments are just a straightforward linear SparseWeights layer.
    It seems to be 3-6 times faster than other implementations depending on settings.
    """

    def __init__(
        self, module, dim_context, module_sparsity, dendrite_sparsity,
        num_segments=1, dendrite_bias=False
    ):
        """
        :param module: linear module from in-units to out-units
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        :param num_segments: number of dendrite segments per out-unit. Must be 1.
        :param dendrite_bias: bool indicating whether or not dendrite activations have
               an additive bias
        """
        assert(num_segments == 1)

        self.dim_context = dim_context
        self.segments = None

        super().__init__(module,
                         sparsity=module_sparsity,
                         allow_extremes=True)

        self.segments = SparseWeights(
            torch.nn.Linear(dim_context,
                            module.weight.shape[0],
                            bias=dendrite_bias),
            sparsity=dendrite_sparsity,
            allow_extremes=True
        )

        self.rezero_weights()

    def rezero_weights(self):
        """Set the previously selected weights to zero."""
        super().rezero_weights()
        if self.segments is not None:  # only none at beginning of init
            self.segments.rezero_weights()

    def forward(self, x, context):
        """Compute of linear layer and apply output of dendrite segments."""
        y = super().forward(x)
        dendrite_activations = self.segments(context)
        return self.apply_dendrites(y, dendrite_activations)

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a sigmoidal gating mechanism."""
        return y * torch.sigmoid(dendrite_activations)

    @property
    def segment_weights(self):
        return self.segments.module.weight

class BiasingDendriticLayer(DendriticLayerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_bias = DendriticBias1d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a bias."""
        return self.dendritic_bias(y, dendrite_activations).values


class AbsoluteMaxGatingDendriticLayer(DendriticLayerBase):
    """
    This layer is similar to `GatingDendriticLayer`, but selects dendrite activations
    based on absolute max activation values instead of just max activation values. For
    example, if choosing between activations -7.4, and 6.5 for a particular unit, -7.4
    will be chosen, and its sign will be kept.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendritic_absolute_max_gate = DendriticAbsoluteMaxGate1d()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        return self.dendritic_absolute_max_gate(y, dendrite_activations).values


class AbsoluteMaxGatingUnsignedDendriticLayer(DendriticLayerBase):
    """
    This layer performs abs max gating (unsigned).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""

        return (y * torch.sigmoid(dendrite_activations.abs().amax(dim=2)))


class MaxGatingDendriticLayer(DendriticLayerBase):
    """
    This layer performs max gating.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""

        return (y * torch.sigmoid(dendrite_activations.amax(dim=2)))


class FFLayer(SparseWeights):
    """
    Class for a layer of units with no dendritic segments per unit. This is identical
    to a normal feed-forward layer, but useful for debugging to ensure we use the same
    code paths and that everything else is identical.
    """

    def __init__(self, module, module_sparsity):
        """
        :param module: linear module from in-units to out-units
        :param module_sparsity: sparsity applied over linear module
        """
        super().__init__(module,
                         sparsity=module_sparsity,
                         allow_extremes=True)

        self.rezero_weights()

    def forward(self, x, context):
        return super().forward(x)
