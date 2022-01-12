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

from copy import deepcopy
from collections import defaultdict
import torch
from copy import deepcopy

from main.hooks.sparse_viz import (
    AverageSegmentActivationsHook,
    HiddenActivationsPercentOnHook,
    CombinedSparseVizHook
)

from .base import multiseg_mt10_base


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

seg5 = dict(
    num_segments=5,
)
seg5.update(multiseg_mt10_base)


'''
Round 1 of experiments
'''

baseline_5d = deepcopy(seg5)
baseline_5d.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(350, 350),
    kw_percent_on=0.25,
    weight_sparsity=0.5,
    fp16=True,
    preprocess_output_dim=32,
)

baseline_5d_bigger = deepcopy(seg5)
baseline_5d_bigger.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1000, 1000),
    kw_percent_on=0.25,
    weight_sparsity=0.75,
    fp16=True,
    preprocess_output_dim=32,
)

baseline_10d = deepcopy(seg10)
baseline_10d.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(232, 232),
    kw_percent_on=0.25,
    weight_sparsity=0.5,
    fp16=True,
    preprocess_output_dim=32,
)


'''
Round 2 of experiments
'''

no_overlap_10d = deepcopy(seg10)
no_overlap_10d.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_unsigned",
)

overlap_input_10d = deepcopy(seg10)
overlap_input_10d.update(
    input_data="obs|context",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_unsigned",
)

overlap_context_10d = deepcopy(seg10)
overlap_context_10d.update(
    input_data="obs",
    context_data="obs|context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1600, 1600),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=20,
    dendritic_layer_class="abs_max_gating_unsigned",
)

overlap_both_10d = deepcopy(seg10)
overlap_both_10d.update(
    input_data="obs|context",
    context_data="obs|context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1600, 1600),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=20,
    dendritic_layer_class="abs_max_gating_unsigned",
)

'''
Sanity checking Round 2 results
'''

no_overlap_10d_preprocess_relu = deepcopy(seg10)
no_overlap_10d_preprocess_relu.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="max_gating",
)

no_overlap_10d_preprocess_kw = deepcopy(seg10)
no_overlap_10d_preprocess_kw.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_module_type="kw",
    preprocess_output_dim=10,
    dendritic_layer_class="max_gating",
)

no_overlap_10d_preprocess_none = deepcopy(seg10)
no_overlap_10d_preprocess_none.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_module_type=None,
    preprocess_output_dim=10,
    dendritic_layer_class="max_gating",
)


no_overlap_10d_abs_max_unsigned = deepcopy(seg10)
no_overlap_10d_abs_max_unsigned.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_unsigned",
)

no_overlap_15d_abs_max_unsigned = deepcopy(seg10)
no_overlap_15d_abs_max_unsigned.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    num_segments=15,
    preprocess_module_type="relu",
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_unsigned",
)

no_overlap_10d_abs_max_signed = deepcopy(seg10)
no_overlap_10d_abs_max_signed.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Paper Figures",
    evaluation_frequency=5,
)

dendrites_relu = deepcopy(seg10)
dendrites_relu.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1950, 1950),
    layers_modulated=(1,),
    kw_percent_on=0.0,
    weight_sparsity=0.90,
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
)

###

dendrite_test1 = deepcopy(seg10)
dendrite_test1.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2000, 2000),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.2),
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Experimentation",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)

dendrite_test2 = deepcopy(seg10)
dendrite_test2.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(3000, 3000),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.65),
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Experimentation",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)

dendrite_test3 = deepcopy(seg10)
dendrite_test3.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(3500, 3500),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.47),
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Experimentation",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)

dendrite_test4 = deepcopy(seg10)
dendrite_test4.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(3500, 3500),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.65),
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Experimentation",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)


dendrite_test5 = deepcopy(seg10)
dendrite_test5.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(3000, 3000),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.80),
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Final",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
)

no_preprocess1 = deepcopy(seg10)
no_preprocess1.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(3000, 3000),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.65),
    fp16=True,
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Experimentation",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)

no_preprocess2 = deepcopy(seg10)
no_preprocess2.update(
    input_data="obs",
    context_data="context",
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(3000, 3000),
    layers_modulated=(1,),
    kw_percent_on=0.25,
    weight_sparsity=(1-0.80),
    fp16=True,
    preprocess_module_type=None,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Final",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
)

no_preprocess3 = deepcopy(seg10)
no_preprocess3.update(
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
    wandb_group="MT10: Abhi Final",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=320443,
)

no_preprocess4 = deepcopy(seg10)
no_preprocess4.update(
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
    wandb_group="MT10: Abhi Final",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=528491,
)

no_preprocess5 = deepcopy(seg10)
no_preprocess5.update(
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
    wandb_group="MT10: Abhi Final",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=301402,
)

no_preprocess6 = deepcopy(seg10)
no_preprocess6.update(
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
    wandb_group="MT10: Abhi Final",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=123456,
)

no_preprocess7 = deepcopy(seg10)
no_preprocess7.update(
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
    wandb_group="MT10: Abhi Final",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=888007,
)


no_preprocess_final = deepcopy(seg10)
no_preprocess_final.update(
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
    wandb_group="Strategy 1",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
)


CONFIGS = dict(
    baseline_5d=baseline_5d,
    baseline_10d=baseline_10d,
    baseline_5d_bigger=baseline_5d_bigger,
    no_overlap_10d=no_overlap_10d,
    overlap_input_10d=overlap_input_10d,
    overlap_context_10d=overlap_context_10d,
    overlap_both_10d=overlap_both_10d,
    no_overlap_10d_preprocess_relu=no_overlap_10d_preprocess_relu,
    no_overlap_10d_preprocess_kw=no_overlap_10d_preprocess_kw,
    no_overlap_10d_preprocess_none=no_overlap_10d_preprocess_none,
    no_overlap_10d_abs_max_unsigned=no_overlap_10d_abs_max_unsigned,
    no_overlap_15d_abs_max_unsigned=no_overlap_15d_abs_max_unsigned,
    no_overlap_10d_abs_max_signed=no_overlap_10d_abs_max_signed,
    dendrites_relu=dendrites_relu,
    dendrite_test1=dendrite_test1,
    dendrite_test2=dendrite_test2,
    dendrite_test3=dendrite_test3,
    dendrite_test4=dendrite_test4,
    dendrite_test5=dendrite_test5,
    no_preprocess1=no_preprocess1,
    no_preprocess2=no_preprocess2,
    no_preprocess3=no_preprocess3,
    no_preprocess4=no_preprocess4,
    no_preprocess5=no_preprocess5,
    no_preprocess6=no_preprocess6,
    no_preprocess7=no_preprocess7,
    no_preprocess_final=no_preprocess_final,
)