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
import numpy as np
from garage.torch.policies.stochastic_policy import StochasticPolicy
import torch
from torch import nn
from torch.distributions import Normal

from main.modules import (
    GaussianTwoHeadedMLPModule,
    GaussianTwoHeadedDendriticMLPModule,
)

def get_input_separate(observations, num_tasks):
    return observations[:, :-num_tasks]

def get_context_separate(observations, num_tasks):
    return observations[:, -num_tasks:]

def get_mixed_data(observations, num_tasks):
    return torch.cat([observations[:, :-num_tasks], observations[:, -num_tasks:]], 1)


class GaussianMLPPolicy(StochasticPolicy):
    """Multiheaded MLP whose outputs are fed into a Normal distribution.
    A policy that contains a MLP to make prediction based on a gaussian
    distribution.
    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(
        self,
        env_spec,
        hidden_sizes,
        hidden_nonlinearity,
        output_nonlinearity,
        min_std,
        max_std,
        normal_distribution_cls,
        init_std=1.0,
        std_parameterization="exp",
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
        learn_std=True
    ):
        super().__init__(env_spec, name="GaussianPolicy")

        self.module = GaussianTwoHeadedMLPModule(
            input_dim=env_spec.observation_space.flat_dim,
            output_dim=env_spec.action_space.flat_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            min_std=min_std,
            max_std=max_std,
            normal_distribution_cls=normal_distribution_cls,
            init_std=init_std,
            std_parameterization=std_parameterization,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
            learn_std=learn_std
        )


    def forward(self, observations):
        """Compute the action distributions from the observations.
        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
        """
        dist = self.module(observations)

        return dist, dict(mean=dist.mean, log_std=(dist.variance.sqrt()).log())


class GaussianDendriticMLPPolicy(StochasticPolicy):
    """Multiheaded Dendritic MLP whose outputs are fed into a Normal distribution.
    A policy that contains a MLP to make prediction based on a gaussian
    distribution.
    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(
        self,
        env_spec,
        num_tasks,
        input_data,
        context_data,
        hidden_sizes,
        layers_modulated,
        num_segments,
        kw_percent_on,
        context_percent_on,
        weight_sparsity,
        weight_init,
        dendrite_weight_sparsity,
        dendrite_init,
        dendritic_layer_class,
        output_nonlinearity,
        preprocess_module_type,
        preprocess_output_dim,
        preprocess_kw_percent_on,
        min_std,
        max_std,
        normal_distribution_cls,
        init_std=1.0,
        std_parameterization="exp",
        layer_normalization=False,
        learn_std=True
    ):
        super().__init__(env_spec, name="GaussianPolicy")

        self.num_tasks = num_tasks

        self.input_func = None
        self.context_func = None

        if input_data == "obs":
            self.input_dim = env_spec.observation_space.flat_dim - self.num_tasks
            self.input_func = get_input_separate
        elif input_data == "obs|context":
            self.input_dim = env_spec.observation_space.flat_dim
            self.input_func = get_mixed_data

        if context_data == "context":
            self.context_dim = self.num_tasks
            self.context_func = get_context_separate
        elif context_data == "obs|context":
            self.context_dim = env_spec.observation_space.flat_dim
            self.context_func = get_mixed_data

        self.module = GaussianTwoHeadedDendriticMLPModule(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_dim=env_spec.action_space.flat_dim,
            hidden_sizes=hidden_sizes,
            layers_modulated=layers_modulated,
            num_segments=num_segments,
            kw_percent_on=kw_percent_on,
            context_percent_on=context_percent_on,
            weight_sparsity=weight_sparsity,
            weight_init=weight_init,
            dendrite_weight_sparsity=dendrite_weight_sparsity,
            dendrite_init=dendrite_init,
            dendritic_layer_class=dendritic_layer_class,
            output_nonlinearity=output_nonlinearity,
            preprocess_module_type=preprocess_module_type,
            preprocess_output_dim=preprocess_output_dim,
            preprocess_kw_percent_on=preprocess_kw_percent_on,
            min_std=min_std,
            max_std=max_std,
            normal_distribution_cls=normal_distribution_cls,
            init_std=init_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            learn_std=learn_std
        )


    def forward(self, observations):
        """Compute the action distributions from the observations.
        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
        """
        obs_portion = self.input_func(observations=observations, num_tasks=self.num_tasks)
        context_portion = self.context_func(observations=observations, num_tasks=self.num_tasks)

        dist = self.module(obs_portion, context_portion)

        return dist, dict(mean=dist.mean, log_std=(dist.variance.sqrt()).log())