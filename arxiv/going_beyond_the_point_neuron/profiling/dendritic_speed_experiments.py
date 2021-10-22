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
import time

import torch
import torch.autograd.profiler as profiler

from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    DendriticLayerBase,
    OneSegmentDendriticLayer,
)


class FlatAbsoluteMaxLayer(DendriticLayerBase):
    """
    Same as AbsoluteMaxGatingDendriticLayer but removes all the indirection and calls.
    It's about 20% faster than AbsoluteMaxGatingDendriticLayer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_dendrites(self, y, dendrite_activations):
        indices = dendrite_activations.abs().max(dim=2).indices
        unsqueezed = indices.unsqueeze(dim=2)
        dendrite_activations = torch.gather(dendrite_activations, dim=2,
                                            index=unsqueezed)
        winning_activations = dendrite_activations.squeeze(dim=2)
        return y * torch.sigmoid(winning_activations)


class FlatOneSegmentLayer(DendriticLayerBase):
    """
    Like FlatAbsoluteMaxLayer but assumes one segment per unit. This is about 20%
    faster than FlatAbsoluteMaxLayer and 45% faster than AbsoluteMaxGatingDendriticLayer
    """

    def __init__(self, *args, **kwargs):
        """Assumes exactly one segment"""
        super().__init__(*args, **kwargs)
        assert(self.segments.num_segments == 1)

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        winning_activations = dendrite_activations.squeeze(dim=2)
        return y * torch.sigmoid(winning_activations)


def profile_model(device,
                  input_size=10,
                  num_units=10,
                  num_segments=20,
                  dim_context=15,
                  batch_size=4096,
                  iterations=10,
                  dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):
    """Create dendritic layer using the specified layer type, and profile it."""

    print("\n\n=============== " + dendritic_layer_class.__name__
          + " ================")
    use_cuda = device.type == "cuda"
    linear = torch.nn.Linear(input_size, num_units)
    dendrite_layer = dendritic_layer_class(
        module=linear,
        num_segments=num_segments,
        dim_context=dim_context,
        module_sparsity=0.7,
        dendrite_sparsity=0.9
    ).to(device)

    dummy_tensor = torch.rand((batch_size, input_size), device=device)
    dummy_context = torch.rand((batch_size, dim_context), device=device)

    s = time.time()
    with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
        with profiler.record_function(dendritic_layer_class.__name__ + " inference"):
            res = dendrite_layer(dummy_tensor, dummy_context)
            for _ in range(iterations - 1):
                res += dendrite_layer(dummy_tensor, dummy_context)
    wall_clock = time.time() - s
    print("Wall clock:", wall_clock)

    if device.type == "cuda":
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if res.sum() == 0:  # Just to make Python think we need res
        print(res.sum())

    return wall_clock


def run_basic_profile():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Comparisons with 20 segments per unit
    for dendrite_class in [
        AbsoluteMaxGatingDendriticLayer,
        FlatAbsoluteMaxLayer,
    ]:
        profile_model(device,
                      input_size=100,
                      dim_context=11,
                      num_units=102,
                      dendritic_layer_class=dendrite_class)

    # Comparisons with one segment per unit
    for dendrite_class in [
        AbsoluteMaxGatingDendriticLayer,
        FlatAbsoluteMaxLayer,
        FlatOneSegmentLayer,
        OneSegmentDendriticLayer,
    ]:
        profile_model(device,
                      input_size=100,
                      dim_context=100,
                      num_units=2048,
                      num_segments=1,
                      iterations=5,
                      dendritic_layer_class=dendrite_class)


if __name__ == "__main__":
    run_basic_profile()
