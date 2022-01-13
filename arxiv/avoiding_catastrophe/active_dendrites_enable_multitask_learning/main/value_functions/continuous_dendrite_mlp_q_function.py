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
"""This modules creates a continuous Q-function network with a Dendrite MLP."""

import torch

from main.modules import CustomDendriticMLP

def get_input_separate(observations, actions, num_tasks):
    return torch.cat([observations[:, :-num_tasks], actions], 1)

def get_context_separate(observations, actions, num_tasks):
    return observations[:, -num_tasks:]

def get_mixed_data(observations, actions, num_tasks):
    return torch.cat([observations[:, :-num_tasks], actions, observations[:, -num_tasks:]], 1)


class ContinuousDendriteMLPQFunction(CustomDendriticMLP):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
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
        preprocess_output_dim
    ):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        self.num_tasks = num_tasks

        self.input_func = None
        self.context_func = None

        if input_data == "obs":
            self.input_dim = env_spec.observation_space.flat_dim - self.num_tasks + env_spec.action_space.flat_dim
            self.input_func = get_input_separate
        elif input_data == "obs|context":
            self.input_dim = env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim
            self.input_func = get_mixed_data

        if context_data == "context":
            self.context_dim = self.num_tasks
            self.context_func = get_context_separate
        elif context_data == "obs|context":
            self.context_dim = env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim
            self.context_func = get_mixed_data


        super().__init__(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            output_sizes=1,
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
            preprocess_kw_percent_on=kw_percent_on
        )

    def forward(self, observations, actions):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.

        Returns:
            torch.Tensor: Output value
        """
        obs_portion = self.input_func(observations=observations, actions=actions, num_tasks=self.num_tasks)
        context_portion = self.context_func(observations=observations, actions=actions, num_tasks=self.num_tasks)

        return super().forward(obs_portion, context_portion)
