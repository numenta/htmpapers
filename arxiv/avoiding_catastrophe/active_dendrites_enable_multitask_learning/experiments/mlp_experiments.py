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

from copy import deepcopy

'''
Round 1 of experiments
'''

metaworld_base = deepcopy(mlp_mt10_base)
metaworld_base.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(400, 400),
    kw_percent_on=None,
    fp16=True,
)

gradient_surgery_base = deepcopy(mlp_mt10_base)
gradient_surgery_base.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(160, 160, 160, 160, 160, 160),
    kw_percent_on=None,
    fp16=True,
)

baseline_similar_metaworld = deepcopy(mlp_mt10_base)
baseline_similar_metaworld.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(580, 580),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=0.5,
)

baseline_similarv2_metaworld = deepcopy(mlp_mt10_base)
baseline_similarv2_metaworld.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1000, 1000),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=0.5,
)


'''
Round 2 of experiments
'''

new_metaworld_baseline = deepcopy(mlp_mt10_base)
new_metaworld_baseline.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(750, 750),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
)


'''
Round 3 of experiments
'''

bigger_metaworld_baseline = deepcopy(mlp_mt10_base)
bigger_metaworld_baseline.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1600, 1600),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
)


mlp_test1 = deepcopy(mlp_mt10_base)
mlp_test1.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2500, 2500),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Experimentation",
    share_train_eval_env=True,
    task_update_frequency=1e12,
)

mlp_test2 = deepcopy(mlp_mt10_base)
mlp_test2.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(1000, 1000),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Experimentation",
    share_train_eval_env=True,
    task_update_frequency=1e12,
)

mlp_test3 = deepcopy(mlp_mt10_base)
mlp_test3.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(3000, 3000),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Experimentation",
    share_train_eval_env=True,
    task_update_frequency=1e12,
)

mlp_test4 = deepcopy(mlp_mt10_base)
mlp_test4.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(3000, 3000),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Final",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
)

mlp_test5 = deepcopy(mlp_mt10_base)
mlp_test5.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Abhi Final",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=401513,
)

mlp_test6 = deepcopy(mlp_mt10_base)
mlp_test6.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Abhi Final",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=123456,
)

mlp_test7 = deepcopy(mlp_mt10_base)
mlp_test7.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Abhi Final",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=607891,
)

mlp_test8 = deepcopy(mlp_mt10_base)
mlp_test8.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="MT10: Abhi Final",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=871291,
)

mlp_test9 = deepcopy(mlp_mt10_base)
mlp_test9.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(2800, 2800),
    kw_percent_on=None,
    fp16=True,
    weight_sparsity=1.0,
    wandb_group="Strategy 2",
    share_train_eval_env=True,
    task_update_frequency=1e12,
    log_per_task=True,
    env_seed=59071,
    params_seed=456789,
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


CONFIGS = dict(
    metaworld_base=metaworld_base,
    gradient_surgery_base=gradient_surgery_base,
    baseline_similar_metaworld=baseline_similar_metaworld,
    baseline_similarv2_metaworld=baseline_similarv2_metaworld,
    new_metaworld_baseline=new_metaworld_baseline,
    bigger_metaworld_baseline=bigger_metaworld_baseline,
    mlp_test1=mlp_test1,
    mlp_test2=mlp_test2,
    mlp_test3=mlp_test3,
    mlp_test4=mlp_test4,
    mlp_test5=mlp_test5,
    mlp_test6=mlp_test6,
    mlp_test7=mlp_test7,
    mlp_test8=mlp_test8,
    mlp_test9=mlp_test9,
    mlp_final=mlp_final,
)