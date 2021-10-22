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

from dendritic_speed_experiments import OneSegmentDendriticLayer
from models import DendriticMLP, SparseMLP
from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.research.frameworks.pytorch.models.common_models import StandardMLP


def func(model, device, input_size, epochs=100, dendrite=False):
    batch_size = 4096
    use_cuda = device.type == "cuda"
    dummy_tensor = torch.rand((batch_size, input_size), device=device)
    wall_clock = 0.0
    for _ in range(epochs):
        if dendrite:
            dummy_context = torch.rand((batch_size, model.dim_context), device=device)

            s = time.time()
            with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
                with profiler.record_function("model_inference"):
                    res = model(dummy_tensor, dummy_context)
        else:
            s = time.time()
            with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
                with profiler.record_function("model_inference"):
                    res = model(dummy_tensor)

        wall_clock += time.time() - s
    print("Wall clock:", wall_clock / epochs)
    if device.type == "cuda":
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    dense_params, sparse_params = count_nonzero_params(model)
    print(f"Total params:{dense_params}, non-zero params:{sparse_params}")

    if res.sum() == 0:  # Just to make Python think we need res
        print(res.sum())

    return wall_clock / epochs


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dim_context = 10
    dendritic_net_ff_input_dim = 11
    non_dendritic_net_input_dim = dendritic_net_ff_input_dim + dim_context
    output_dim = 10
    dendrite_net = DendriticMLP(
        hidden_sizes=(2048, 2048, 2048),
        input_size=dendritic_net_ff_input_dim,
        output_dim=output_dim,
        k_winners=True,
        relu=False,
        k_winner_percent_on=0.1,
        dim_context=dim_context,
        num_segments=(1, 1, 1),
        sparsity=0.5,
        dendritic_layer_class=OneSegmentDendriticLayer
    ).to(device)

    dense_net = StandardMLP(
        input_size=non_dendritic_net_input_dim,
        num_classes=output_dim,
        hidden_sizes=(2048, 2048, 2048),
    ).to(device)

    sparse_net = SparseMLP(
        input_size=non_dendritic_net_input_dim,
        output_dim=output_dim,
        hidden_sizes=(2048, 2048, 2048),
        linear_activity_percent_on=(0.1, 0.1, 0.1),
        linear_weight_percent_on=(0.5, 0.5, 0.5),
        k_inference_factor=1.0,
        use_batch_norm=False,
    ).to(device)

    print("=================== DENSE NETWORK =====================")
    print(dense_net)
    dense_time = func(dense_net, input_size=non_dendritic_net_input_dim, device=device)

    print("\n\n=================== SPARSE NETWORK =====================")
    print(sparse_net)
    sparse_time = func(sparse_net, input_size=non_dendritic_net_input_dim,
                       device=device)

    print("\n\n=================== SPARSE DENDRITIC NETWORK =====================")
    print(dendrite_net)
    dendrite_time = func(dendrite_net, input_size=dendritic_net_ff_input_dim,
                         device=device, dendrite=True)

    print(f"Ratio of sparse to dense: {sparse_time/dense_time}")
    print(f"Ratio of dendritic to dense: {dendrite_time/dense_time}")
