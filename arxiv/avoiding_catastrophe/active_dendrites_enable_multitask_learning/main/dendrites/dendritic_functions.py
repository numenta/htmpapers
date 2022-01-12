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

from collections import namedtuple

import torch
from typing import Optional


dendrite_output = namedtuple("dendrite_output", ["values", "indices"])
dendrite_output.__doc__ = """
A named tuple for outputs modified by `apply_dendrites`_.

:attr values: output tensor after being modulated by dendrite activations
:attr indices: the indices of the winning segments used to modulate the output tensor
"""

def gather_activations(dendrite_activations, indices):
    """
    Gathers dendritic activations from the given indices.
    :param indices: tensor of indices of winning segments;
                    shape of batch_size x num_units
    :param indices: tensor of dendritic activations;
                    shape of batch_size x num_units x num_segments
    """
    unsqueezed = indices.unsqueeze(dim=2)
    dendrite_activations = torch.gather(dendrite_activations, dim=2, index=unsqueezed)
    dendrite_activations = dendrite_activations.squeeze(dim=2)
    return dendrite_activations


def dendritic_bias_1d(y, dendrite_activations):
    """
    Returns the sum of the feedforward output and the max of the dendrite
    activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              size and number of units, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, s) where the
                                 axes represent the batch size, number of units, and
                                 number of segments respectively.
    """
    # Take max along each segment.
    winning_activations, indices = dendrite_activations.max(dim=2)
    return dendrite_output(y + winning_activations, indices)


def dendritic_gate_1d(y, dendrite_activations, indices: Optional[torch.Tensor] = None):
    """
    Returns the product of the feedforward output and sigmoid of the the max
    of the dendrite activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              size and number of units, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, s) where the
                                 axes represent the batch size, number of units, and
                                 number of segments, respectively.
    :param indices: (optional) indices of winning segments;
                    shape of batch_size x num_units
    """
    # Select winner by max activations, or use given indices as winners.
    if indices is None:
        winning_activations, indices = dendrite_activations.max(dim=2)
    else:
        winning_activations = gather_activations(dendrite_activations, indices)

    # Multiple by the sigmoid of the max along each segment.
    return dendrite_output(y * torch.sigmoid(winning_activations), indices)


def dendritic_absolute_max_gate_1d(y, dendrite_activations):
    """
    Returns the product of the feedforward output and the sigmoid of the
    absolute max of the dendrite activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              size and number of units, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, s) where the
                                 axes represent the batch size, number of units, and
                                 number of segments, respectively.
    """
    indices = dendrite_activations.abs().max(dim=2).indices
    return dendritic_gate_1d(y, dendrite_activations, indices=indices)
