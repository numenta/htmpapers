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
Multitask Experiment configuration to test policy data collection hooks
"""

from copy import deepcopy
import torch
from collections import defaultdict

from main.hooks.sparse_viz import (
    AverageSegmentActivationsHook,
    HiddenActivationsPercentOnHook,
    CombinedSparseVizHook
)

from .multiseg_experiments import no_overlap_10d_abs_max_signed, dendrites_relu, seg10
from .mlp_experiments import new_metaworld_baseline
from .base import mlp_mt10_base


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


debug = deepcopy(no_overlap_10d_abs_max_signed)
debug.update(
    evaluation_frequency=1,
    timesteps=100000,
    buffer_batch_size=32,
    num_grad_steps_scale=0.01,
)

test_hook = deepcopy(debug)
test_hook.update(
    policy_data_collection_hook=HookManagerSample,
)

test_sparse_hook = deepcopy(debug)
test_sparse_hook.update(
    policy_data_collection_hook=CombinedSparseVizHook,
    override_weight_initialization=True,
    params_seed=123455
)

no_overlap_10d_abs_max_signed_with_plots = deepcopy(no_overlap_10d_abs_max_signed)
no_overlap_10d_abs_max_signed_with_plots.update(
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)


no_overlap_10d_abs_max_signed_with_plots_mt50 = deepcopy(no_overlap_10d_abs_max_signed_with_plots)
no_overlap_10d_abs_max_signed_with_plots_mt50.update(
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    num_tasks=50,
    cpus_per_worker=0.14
)

dendrites_relu_with_plots = deepcopy(dendrites_relu)
dendrites_relu_with_plots.update(
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
)

dendrites_with_plots_noenvupdate = deepcopy(no_overlap_10d_abs_max_signed)
dendrites_with_plots_noenvupdate.update(
    task_update_frequency=1e12,  # fix it
    share_train_eval_env=True,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=False,  # True
    evaluation_frequency=10,
)

# No sparsity, same size as the baseline
dendrites_with_plots_noenvupdate_samesize = deepcopy(dendrites_with_plots_noenvupdate)
dendrites_with_plots_noenvupdate_samesize.update(
    hidden_sizes=(750, 750),
    weight_sparsity=0.0,
    timesteps=30000000
)


# No sparsity, same size as the baseline
replicate_previous = deepcopy(dendrites_with_plots_noenvupdate)
replicate_previous.update(
    context_data="obs|context",
    preprocess_output_dim=64,
    hidden_sizes=(750, 750),
    weight_sparsity=0.0,
    timesteps=30000000
)


new_metaworld_baseline_noenvupdate = deepcopy(new_metaworld_baseline)
new_metaworld_baseline_noenvupdate.update(
    task_update_frequency=1e12,  # fix it
    share_train_eval_env=True,
    evaluation_frequency=10,
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
    weight_sparsity=(1 - 0.65),
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Paper Figures",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    # override_weight_initialization=True,
    log_per_task=True
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
    weight_sparsity=(1 - 0.80),
    fp16=True,
    preprocess_output_dim=10,
    dendritic_layer_class="abs_max_gating_signed",
    wandb_group="MT10: Final-HPSearch",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
)

hpsearch_lr = [None, 1e-4, 3e-3, 1e-3, 3e-2, 1e-2]

dendrite_test5_lr1 = deepcopy(dendrite_test5)
dendrite_test5_lr1.update(
    policy_lr=hpsearch_lr[1],
    qf_lr=hpsearch_lr[1],
)

dendrite_test5_lr2 = deepcopy(dendrite_test5)
dendrite_test5_lr2.update(
    policy_lr=hpsearch_lr[2],
    qf_lr=hpsearch_lr[2],
)

dendrite_test5_lr3 = deepcopy(dendrite_test5)
dendrite_test5_lr3.update(
    policy_lr=hpsearch_lr[3],
    qf_lr=hpsearch_lr[3],
)


dendrite_test5_lr4 = deepcopy(dendrite_test5)
dendrite_test5_lr4.update(
    policy_lr=hpsearch_lr[4],
    qf_lr=hpsearch_lr[4],
)

dendrite_test5_lr5 = deepcopy(dendrite_test5)
dendrite_test5_lr5.update(
    policy_lr=hpsearch_lr[5],
    qf_lr=hpsearch_lr[5],
)


dendrite_test_90dense = deepcopy(dendrite_test2)
dendrite_test_90dense.update(
    hidden_sizes=(1950, 1950),
    weight_sparsity=0.1,
)


sparse_test2 = deepcopy(dendrite_test2)
sparse_test2.update(
    input_data="obs|context",
    net_type="MLP",
    policy_data_collection_hook=None,
    log_per_task=True,
    share_train_eval_env=False,
    task_update_frequency=1,
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
    wandb_group="MT10: Lucas Final",
    evaluation_frequency=10,
    share_train_eval_env=True,
    task_update_frequency=1e12,
    policy_data_collection_hook=CombinedSparseVizHook,
    save_visualizations_local=True,
    log_per_task=True,
    env_seed=59071,
    params_seed=321654,
)

mlp_test5 = deepcopy(mlp_mt10_base)
mlp_test5.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Lucas Final",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=401513,
)

mlp_test6 = deepcopy(mlp_test5)
mlp_test6.update(
    params_seed=401509,
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

mlp_final = deepcopy(mlp_mt10_base)
mlp_final.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="Strategy 1",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
)

# Export configurations in this file
CONFIGS = dict(
    test_hook=test_hook,
    test_sparse_hook=test_sparse_hook,
    no_overlap_10d_abs_max_signed_with_plots=no_overlap_10d_abs_max_signed_with_plots,
    no_overlap_10d_abs_max_signed_with_plots_mt50=no_overlap_10d_abs_max_signed_with_plots_mt50,
    dendrites_relu_with_plots=dendrites_relu_with_plots,
    new_metaworld_baseline_noenvupdate=new_metaworld_baseline_noenvupdate,
    dendrites_with_plots_noenvupdate=dendrites_with_plots_noenvupdate,
    dendrites_with_plots_noenvupdate_samesize=dendrites_with_plots_noenvupdate_samesize,
    replicate_previous=replicate_previous,
    dendrite_test2=dendrite_test2,
    sparse_test2=sparse_test2,
    dendrite_test_90dense=dendrite_test_90dense,
    dendrite_test5_lr1=dendrite_test5_lr1,
    dendrite_test5_lr2=dendrite_test5_lr2,
    dendrite_test5_lr3=dendrite_test5_lr3,
    dendrite_test5_lr4=dendrite_test5_lr4,
    dendrite_test5_lr5=dendrite_test5_lr5,
    no_preprocess6=no_preprocess6,
    mlp_test5=mlp_test5,
    mlp_test6=mlp_test6,
    no_preprocess_final=no_preprocess_final,
    mlp_final=mlp_final
)