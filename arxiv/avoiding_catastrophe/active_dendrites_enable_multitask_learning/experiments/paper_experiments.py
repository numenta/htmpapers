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

from .base import mlp_mt10_base
from .base import multiseg_mt10_base

from copy import deepcopy
from collections import defaultdict
import torch

from main.hooks.sparse_viz import (
    AverageSegmentActivationsHook,
    HiddenActivationsPercentOnHook,
    CombinedSparseVizHook
)

class HookManagerSample:
    """
    Requires:
    - assigning a function to collect_hook_data in the recipient network
    - attaching a hook to the recipient network
    - a class method called consolidate_and_report that executes an action
    based on the data reported
    """

    def __init__(self, network):
        self.hook_data = []
        # redirect function to the network
        network.collect_hook_data = self.export_data
        # attach hook
        network.module.mean_log_std.register_forward_hook(
            self.forward_hook
        )

    def forward_hook(self, m, i, o):
        self.hook_data.append(i[0][0])

    def export_data(self):
        """Returns current data and reinitializes collection"""
        data_to_export = self.hook_data
        self.hook_data = []
        return data_to_export

    @classmethod
    def consolidate_and_report(cls, data):
        """
        Accepts a dictionary where key is the task index
        and value is a list with one entry per step take

        Class method, requires data argument

        Returns a dictionary that can be incorporated into a regular log dict
        """
        sum_inputs_per_task = defaultdict(int)
        for task_id, task_data in data.items():
            for step_data in task_data:
                sum_inputs_per_task[task_id] += torch.sum(step_data).item()
        print(sum_inputs_per_task)

        return {"sum_inputs_per_task": sum_inputs_per_task.values()}


seg10 = dict(
    num_segments=10,
)
seg10.update(multiseg_mt10_base)


'''
GENERAL NOTE: ALL RUNS USE A RANDOM ENVIRONMENT SEED (SOME RANDOMLY FIXED GOAL)
'''


'''
MLP RUNS:

5 configs that have been seeded with random parameter seeds
1 config that uses a random parameter seed each time
'''
mlp_run_seed1 = deepcopy(mlp_mt10_base)
mlp_run_seed1.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="Paper Figures",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=401513,
)

mlp_run_seed2 = deepcopy(mlp_mt10_base)
mlp_run_seed2.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="Paper Figures",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=123456,
)

mlp_run_seed3 = deepcopy(mlp_mt10_base)
mlp_run_seed3.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="Paper Figures",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=607891,
)

mlp_run_seed4 = deepcopy(mlp_mt10_base)
mlp_run_seed4.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="Paper Figures",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=871291,
)

mlp_run_seed5 = deepcopy(mlp_mt10_base)
mlp_run_seed5.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="Paper Figures",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=456789,
)


mlp_run_random_seed = deepcopy(mlp_mt10_base)
mlp_run_random_seed.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="Paper Figures",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
)

'''
DENDRITE RUNS:

5 configs that have been seeded with random parameter seeds
1 config that uses a random parameter seed each time
'''
dendrite_run_seed1 = deepcopy(seg10)
dendrite_run_seed1.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.90),
    fp16=True,
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="Paper Figures",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=320443,
)

dendrite_run_seed2 = deepcopy(seg10)
dendrite_run_seed2.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.90),
    fp16=True,
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="Paper Figures",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=528491,
)

dendrite_run_seed3 = deepcopy(seg10)
dendrite_run_seed3.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.90),
    fp16=True,
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="Paper Figures",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=301402,
)

dendrite_run_seed4 = deepcopy(seg10)
dendrite_run_seed4.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.90),
    fp16=True,
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="Paper Figures",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=123456,
)

dendrite_run_seed5 = deepcopy(seg10)
dendrite_run_seed5.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.90),
    fp16=True,
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="Paper Figures",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=888007,
)

dendrite_run_random_seed = deepcopy(seg10)
dendrite_run_random_seed.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.90),
    fp16=True,
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Experimentation",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
)

CONFIGS = dict(
    mlp_run_seed1=mlp_run_seed1,
    mlp_run_seed2=mlp_run_seed2,
    mlp_run_seed3=mlp_run_seed3,
    mlp_run_seed4=mlp_run_seed4,
    mlp_run_seed5=mlp_run_seed5,
    mlp_run_random_seed=mlp_run_random_seed,
    dendrite_run_seed1=dendrite_run_seed1,
    dendrite_run_seed2=dendrite_run_seed2,
    dendrite_run_seed3=dendrite_run_seed3,
    dendrite_run_seed4=dendrite_run_seed4,
    dendrite_run_seed5=dendrite_run_seed5,
    dendrite_run_random_seed=dendrite_run_random_seed,
)