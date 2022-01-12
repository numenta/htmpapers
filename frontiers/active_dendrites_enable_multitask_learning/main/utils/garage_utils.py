# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from dowel import tabular
from garage import EpisodeBatch, StepType
from garage.np import discount_cumsum
from garage.torch import global_device
from garage.torch.distributions import TanhNormal
from garage.torch.q_functions import ContinuousMLPQFunction
from torch.distributions import Normal

from main.policies import (
    GaussianDendriticMLPPolicy,
    GaussianMLPPolicy,
)
from main.value_functions import ContinuousDendriteMLPQFunction

from main.dendrites import MaxGatingDendriticLayer, AbsoluteMaxGatingUnsignedDendriticLayer

from main.dendrites import (
    BiasingDendriticLayer,
    OneSegmentDendriticLayer,
)

from main.dendrites import AbsoluteMaxGatingDendriticLayer as AbsoluteMaxGatingSignedDendriticLayer

import logging

def create_policy_net(env_spec, net_params):
    if net_params.net_type == "MLP":
        net = GaussianMLPPolicy(
            env_spec=env_spec,
            hidden_sizes=net_params.hidden_sizes,
            hidden_nonlinearity=create_nonlinearity(net_params.policy_hidden_nonlinearity),
            output_nonlinearity=create_nonlinearity(net_params.output_nonlinearity),
            min_std=net_params.policy_min_log_std,
            max_std=net_params.policy_max_log_std,
            normal_distribution_cls=create_distribution(net_params.distribution)
        )
    elif net_params.net_type == "Dendrite_MLP":
        dendritic_layer_class = create_dendritic_layer(net_params.dendritic_layer_class)

        net = GaussianDendriticMLPPolicy(
            env_spec=env_spec,
            num_tasks=net_params.num_tasks,
            input_data=net_params.input_data,
            context_data=net_params.context_data,
            hidden_sizes=net_params.hidden_sizes,
            layers_modulated=net_params.layers_modulated,
            num_segments=net_params.num_segments,
            kw_percent_on=net_params.kw_percent_on,
            context_percent_on=net_params.context_percent_on,
            weight_sparsity=net_params.weight_sparsity,
            weight_init=net_params.weight_init,
            dendrite_weight_sparsity=net_params.dendrite_weight_sparsity,
            dendrite_init=net_params.dendrite_init,
            dendritic_layer_class=dendritic_layer_class,
            output_nonlinearity=net_params.output_nonlinearity,
            preprocess_module_type=net_params.preprocess_module_type,
            preprocess_output_dim=net_params.preprocess_output_dim,
            preprocess_kw_percent_on=net_params.kw_percent_on,
            min_std=net_params.policy_min_log_std,
            max_std=net_params.policy_max_log_std,
            normal_distribution_cls=create_distribution(net_params.distribution)
        )
    else:
        raise NotImplementedError

    return net

def create_qf_net(env_spec, net_params):
    if net_params.net_type == "MLP":
        net = ContinuousMLPQFunction(
            env_spec=env_spec,
            hidden_sizes=net_params.hidden_sizes,
            hidden_nonlinearity=create_nonlinearity(net_params.qf_hidden_nonlinearity),
            output_nonlinearity=create_nonlinearity(net_params.output_nonlinearity),
        )
    elif net_params.net_type == "Dendrite_MLP":
        dendritic_layer_class = create_dendritic_layer(net_params.dendritic_layer_class)

        net = ContinuousDendriteMLPQFunction(
            env_spec=env_spec,
            num_tasks=net_params.num_tasks,
            input_data=net_params.input_data,
            context_data=net_params.context_data,
            hidden_sizes=net_params.hidden_sizes,
            layers_modulated=net_params.layers_modulated,
            num_segments=net_params.num_segments,
            kw_percent_on=net_params.kw_percent_on,
            context_percent_on=net_params.context_percent_on,
            weight_sparsity=net_params.weight_sparsity,
            weight_init=net_params.weight_init,
            dendrite_weight_sparsity=net_params.dendrite_weight_sparsity,
            dendrite_init=net_params.dendrite_init,
            dendritic_layer_class=dendritic_layer_class,
            output_nonlinearity=net_params.output_nonlinearity,
            preprocess_module_type=net_params.preprocess_module_type,
            preprocess_output_dim=net_params.preprocess_output_dim,
        )
    else:
        raise NotImplementedError

    return net


def create_dendritic_layer(dendritic_layer):
    if dendritic_layer == "biasing":
        return BiasingDendriticLayer
    elif dendritic_layer == "max_gating":
        return MaxGatingDendriticLayer
    elif dendritic_layer == "abs_max_gating_signed":
        return AbsoluteMaxGatingSignedDendriticLayer
    elif dendritic_layer == "abs_max_gating_unsigned":
        return AbsoluteMaxGatingUnsignedDendriticLayer
    elif dendritic_layer == "one_segment":
        return OneSegmentDendriticLayer
    else:
        raise NotImplementedError


def create_nonlinearity(nonlinearity):
    if nonlinearity == "tanh":
        return torch.tanh
    elif nonlinearity == "relu":
        return torch.relu
    elif nonlinearity == None:
        return None
    else:
        raise NotImplementedError


def get_params(file_name):
    with open(file_name) as f:
        params = json.load(f)
    return params


def create_distribution(distribution):
    if distribution == "Normal":
        return Normal
    elif distribution == "TanhNormal":
        return TanhNormal
    else:
        raise NotImplementedError


def log_multitask_performance(
    itr, batch, discount, name_map=None, log_per_task=False,
):
    r"""Log performance of episodes from multiple tasks.

    Args:
        itr (int): Iteration number to be logged.
        batch (EpisodeBatch): Batch of episodes. The episodes should have
            either the "task_name" or "task_id" `env_infos`. If the "task_name"
            is not present, then `name_map` is required, and should map from
            task id's to task names.
        discount (float): Discount used in computing returns.
        name_map (dict[int, str] or None): Mapping from task id"s to task
            names. Optional if the "task_name" environment info is present.
            Note that if provided, all tasks listed in this map will be logged,
            even if there are no episodes present for them.

    Returns:
        numpy.ndarray: Undiscounted returns averaged across all tasks. Has
            shape :math:`(N \bullet [T])`.

    """

    # Create log_dict with the averages
    undiscounted_returns, consolidated_log = log_performance(
        itr, batch, discount=discount, prefix="Average"
    )

    # Add results by task to task _dict, if requested
    if log_per_task:
        eps_by_name = defaultdict(list)
        for eps in batch.split():
            task_name = "__unnamed_task__"
            if "task_name" in eps.env_infos:
                task_name = eps.env_infos["task_name"][0]
            elif "task_id" in eps.env_infos:
                name_map = {} if name_map is None else name_map
                task_id = eps.env_infos["task_id"][0]
                task_name = name_map.get(task_id, "Task #{}".format(task_id))
            eps_by_name[task_name].append(eps)

        if name_map is None:
            task_names = eps_by_name.keys()
        else:
            task_names = name_map.values()

        for task_name in task_names:
            if task_name in eps_by_name:
                episodes = eps_by_name[task_name]
                # analyze statistics per task
                _, task_log = log_performance(
                    itr,
                    EpisodeBatch.concatenate(*episodes),
                    discount,
                    prefix=task_name,  # specific to task
                )
                consolidated_log.update(task_log)

    return undiscounted_returns, consolidated_log


def log_performance(itr, batch, discount, prefix="Evaluation"):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm"s property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    termination = []
    success = []
    rewards = []
    grasp_success = []
    near_object = []
    episode_mean_grasp_reward = []
    episode_max_grasp_reward = []
    episode_min_grasp_reward = []
    episode_mean_in_place_reward = []
    episode_max_in_place_reward = []
    episode_min_in_place_reward = []
    for eps in batch.split():
        rewards.append(eps.rewards)
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if "success" in eps.env_infos:
            success.append(float(eps.env_infos["success"].any()))
        if "grasp_success" in eps.env_infos:
            grasp_success.append(float(eps.env_infos["grasp_success"].any()))
        if "near_object" in eps.env_infos:
            near_object.append(float(eps.env_infos["near_object"].any()))
        if "grasp_reward" in eps.env_infos:
            episode_mean_grasp_reward.append(
                np.mean(eps.env_infos["grasp_reward"]))
            episode_max_grasp_reward.append(max(eps.env_infos["grasp_reward"]))
            episode_min_grasp_reward.append(min(eps.env_infos["grasp_reward"]))
        if "in_place_reward" in eps.env_infos:
            episode_mean_in_place_reward.append(
                np.mean(eps.env_infos["in_place_reward"]))
            episode_max_in_place_reward.append(
                max(eps.env_infos["in_place_reward"]))
            episode_min_in_place_reward.append(
                min(eps.env_infos["in_place_reward"]))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    log_dict = {}
    log_dict[prefix + "/Iteration"] = itr
    log_dict[prefix + "/NumEpisodes"] = len(returns)
    log_dict[prefix + "/MinReward"] = np.min(rewards)
    log_dict[prefix + "/MaxReward"] = np.max(rewards)
    log_dict[prefix + "/AverageDiscountedReturn"] = average_discounted_return
    log_dict[prefix + "AverageReturn"] = np.mean(undiscounted_returns)
    log_dict[prefix + "/StdReturn"] = np.std(undiscounted_returns)
    log_dict[prefix + "/MaxReturn"] = np.max(undiscounted_returns)
    log_dict[prefix + "/MinReturn"] = np.min(undiscounted_returns)
    log_dict[prefix + "/TerminationRate"] = np.mean(termination)

    if success:
        log_dict[prefix + "/SuccessRate"] = np.mean(success)
    if grasp_success:
        log_dict[prefix + "Misc/GraspSuccessRate"] = np.mean(grasp_success)
    if near_object:
        log_dict[prefix + "Misc/NearObject"] = np.mean(near_object)
    if episode_mean_grasp_reward:
        log_dict[prefix + "Misc/EpisodeMeanGraspReward"] = np.mean(episode_mean_grasp_reward)
        log_dict[prefix + "Misc/EpisodeMeanMaxGraspReward"] = np.mean(episode_max_grasp_reward)
        log_dict[prefix + "Misc/EpisodeMeanMinGraspReward"] = np.mean(episode_min_grasp_reward)
    if episode_mean_in_place_reward:
        log_dict[prefix + "Misc/EpisodeMeanInPlaceReward"] = np.mean(episode_mean_grasp_reward)
        log_dict[prefix + "Misc/EpisodeMeanMaxInPlaceReward"] = np.mean(episode_max_in_place_reward)
        log_dict[prefix + "Misc/EpisodeMeanMinInPlaceReward"] = np.mean(episode_min_in_place_reward)

    return undiscounted_returns, log_dict

def calculate_mean_param(name, network):
    """Calculate and output mean of tensor means for a given network"""
    logging.warn("Logging few samples from the network")
    for param in network.parameters():
        if len(param.size()) == 2:
            logging.warn(param[0, 0].item())
