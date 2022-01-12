#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Base Multitask Experiment configuration.
"""

from copy import deepcopy

base = dict()

debug = deepcopy(base)
debug = dict(
    evaluation_frequency=1,
    timesteps=100000,
    buffer_batch_size=32,
    num_grad_steps_scale=0.01,  # 5 steps,
    # project_id="shqr401",
)

singleseg_mt10_base = dict(
    num_tasks=10,
    net_type="Dendrite_MLP",
    num_segments=1,
    cpus_per_worker=0.5,
    gpus_per_worker=0,
    wandb_group="MT10 - SingleSeg",
)

multiseg_mt10_base = dict(
    num_tasks=10,
    net_type="Dendrite_MLP",
    dendritic_layer_class="max_gating",
    cpus_per_worker=0.5,
    gpus_per_worker=0,
    wandb_group="MT10 - MultiSeg",
)

mlp_mt10_base = dict(
    num_tasks=10,
    net_type="MLP",
    cpus_per_worker=0.5,
    gpus_per_worker=0,
    wandb_group="MT10 - MLP",
)


# Export configurations in this file
CONFIGS = dict(
    base=base,
    debug=debug,
    singleseg_mt10_base=singleseg_mt10_base,
    multiseg_mt10_base=multiseg_mt10_base,
    mlp_mt10_base=mlp_mt10_base,
)