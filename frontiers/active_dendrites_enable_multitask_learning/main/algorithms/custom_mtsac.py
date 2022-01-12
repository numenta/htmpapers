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
import copy
import logging
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from garage import StepType
from garage.torch import as_torch_dict
from garage.torch.algos.mtsac import MTSAC
from torch.cuda.amp import GradScaler, autocast

from main.utils.garage_utils import log_multitask_performance
from nupic.torch.modules.sparse_weights import rezero_weights


class CustomMTSAC(MTSAC):
    def __init__(
        self,
        policy,
        qf1,
        qf2,
        replay_buffer,
        env_spec,
        sampler,
        train_task_sampler,
        *,
        num_tasks,
        gradient_steps_per_itr,
        task_update_frequency=1,
        max_episode_length_eval=None,
        fixed_alpha=None,
        target_entropy=None,
        initial_log_entropy=0.,
        discount=0.99,
        buffer_batch_size=64,
        min_buffer_size=10000,
        target_update_tau=5e-3,
        policy_lr=3e-4,
        qf_lr=3e-4,
        reward_scale=1.0,
        optimizer=torch.optim.Adam,
        num_evaluation_episodes=5,
        # added
        fp16=False,
        log_per_task=False,
        share_train_eval_env=False
    ):

        super().__init__(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            env_spec=env_spec,
            sampler=sampler,
            test_sampler=sampler,  # not used, for compatibility
            train_task_sampler=train_task_sampler,
            num_tasks=num_tasks,
            gradient_steps_per_itr=gradient_steps_per_itr,
            max_episode_length_eval=max_episode_length_eval,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            initial_log_entropy=initial_log_entropy,
            discount=discount,
            buffer_batch_size=buffer_batch_size,
            min_buffer_size=min_buffer_size,
            target_update_tau=target_update_tau,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            optimizer=optimizer,
            steps_per_epoch=1,
            num_evaluation_episodes=num_evaluation_episodes,
        )
        self._train_task_sampler = train_task_sampler
        self._task_update_frequency = task_update_frequency
        self._fp16 = fp16
        self._log_per_task = log_per_task
        self._total_envsteps = 0

        # scalers for fp16
        # TODO: don't initialize gradscalers if not using fp16
        # Also don't save and/or restore
        self._gs_qf1 = GradScaler()
        self._gs_qf2 = GradScaler()
        self._gs_policy = GradScaler()
        self._gs_alpha = GradScaler()

        # get updates for evaluation
        self.eval_env_updates = self.resample_environment(force_update=True)
        self.share_train_eval_env = share_train_eval_env
        if self.share_train_eval_env:
            logging.warn("WARNING: Sharing train and eval environments")

        # Fix bug with alpha with optimizer
        self._use_automatic_entropy_tuning = fixed_alpha is None
        if self._use_automatic_entropy_tuning:
            self._alpha_optimizer = optimizer([self._log_alpha], lr=self._policy_lr)

    def state_dict(self):
        return {
            # parameters
            "policy": self.policy.state_dict(),
            "qf1": self._qf1.state_dict(),
            "qf2": self._qf2.state_dict(),
            "target_qf1": self._target_qf1.state_dict(),
            "target_qf2": self._target_qf2.state_dict(),
            "log_alpha": self._log_alpha,

            # scalers
            "gs_qf1": self._gs_qf1.state_dict(),
            "gs_qf2": self._gs_qf2.state_dict(),
            "gs_policy": self._gs_policy.state_dict(),
            "gs_alpha": self._gs_alpha.state_dict(),

            # optimizers
            "policy_optimizer": self._policy_optimizer.state_dict(),
            "qf1_optimizer": self._qf1_optimizer.state_dict(),
            "qf2_optimizer": self._qf2_optimizer.state_dict(),
            "alpha_optimizer": self._alpha_optimizer.state_dict(),

            # other variables
            "replay_buffer": self.replay_buffer,
            "total_envsteps": self._total_envsteps,
        }

    def load_env_state(self, env_state):
        self.eval_env_updates = env_state

    def load_state(self, state):
        # parameters
        self.policy.load_state_dict(state["policy"])
        self._qf1.load_state_dict(state["qf1"])
        self._qf2.load_state_dict(state["qf2"])
        self._target_qf1.load_state_dict(state["target_qf1"])
        self._target_qf2.load_state_dict(state["target_qf2"])
        self._log_alpha.data = state["log_alpha"]

        # scalers
        self._gs_qf1.load_state_dict(state["gs_qf1"])
        self._gs_qf2.load_state_dict(state["gs_qf2"])
        self._gs_policy.load_state_dict(state["gs_policy"])
        self._gs_alpha.load_state_dict(state["gs_alpha"])

        # optimizers
        self._policy_optimizer.load_state_dict(state["policy_optimizer"])
        self._qf1_optimizer.load_state_dict(state["qf1_optimizer"])
        self._qf2_optimizer.load_state_dict(state["qf2_optimizer"])
        self._alpha_optimizer.load_state_dict(state["alpha_optimizer"])

        # other variables
        self.replay_buffer = state["replay_buffer"]
        self._total_envsteps = state["total_envsteps"]

    def get_updated_policy(self, policy_hook=None):
        with torch.no_grad():
            updated_policy = copy.deepcopy(self.policy)
        updated_policy.eval()
        # attach hooks
        if policy_hook:
            policy_hook(updated_policy)

        return updated_policy

    def update_buffer(self, trajectories):
        """Update Buffer"""

        self._total_envsteps += sum(trajectories.lengths)
        path_returns = []
        for path in trajectories.to_list():
            self.replay_buffer.add_path(dict(
                observation=path["observations"],
                action=path["actions"],
                reward=path["rewards"].reshape(-1, 1),
                next_observation=path["next_observations"],
                terminal=np.array([
                    step_type == StepType.TERMINAL
                    for step_type in path["step_types"]
                ]).reshape(-1, 1)
            ))
            path_returns.append(sum(path["rewards"]))

        self.episode_rewards.append(np.mean(path_returns))

    def resample_environment(self, epoch=0, force_update=False):
        """
        TODO: fix env update in sampler

        Intended behavior:
        if epoch % self._task_update_frequency == 0 or force_update:
            return self._train_task_sampler.sample(self._num_tasks)
        """
        # TODO: remove first line to allow force update
        if epoch % self._task_update_frequency == 0 or force_update:
            return self._train_task_sampler.sample(self._num_tasks)

    def run_epoch(self, epoch, env_steps_per_epoch):
        """
        Run one epoch, which is composed of one N sample collections and N training
        steps. Each training step in their turn is composed of M gradient steps of
        batch size B

        Total number of samples used by the algorithm in a epoch is given by N * M * B
        (steps * gradient_steps * batch size)

        Samples collected are only used to update the buffer, and there is no direct
        influence on number of gradient steps or batch size.

        Returns:
            float: The average return in last epoch cycle.

        """
        t0 = time()

        env_updates = (
            self.eval_env_updates if self.share_train_eval_env
            else self.resample_environment(epoch)
        )

        new_trajectories = self._sampler.obtain_samples(
            num_samples=env_steps_per_epoch,
            agent_update=self.get_updated_policy(),
            env_updates=env_updates,
        )
        self.update_buffer(new_trajectories)
        t1 = time()
        total_losses = self.run_step()
        time_to_collect_samples = t1 - t0
        time_to_update_gradient = time() - t1

        log_dict = self._log_statistics(*total_losses)

        # TODO: switch to logger.debug once logger is fixed
        logging.warn(f"Time to collect samples: {time_to_collect_samples:.2f}")
        logging.warn(f"Time to update gradient: {time_to_update_gradient:.2f}")

        return log_dict

    def run_step(self):
        """
        Run one training step, which is composed of M gradient steps

        For M gradients steps:
        - sample a batch from buffer
        - perform one gradient step in all three networks (policy, qf1 and qf2)
        """

        total_losses = [0, 0, 0]
        for _ in range(self._gradient_steps):
            if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
                samples = as_torch_dict(self.replay_buffer.sample_transitions(
                    self._buffer_batch_size
                ))
                policy_loss, qf1_loss, qf2_loss = self.optimize_policy(samples)
                total_losses[0] += policy_loss
                total_losses[1] += qf1_loss
                total_losses[2] += qf2_loss
                self._update_targets()

        # Normalize losses by total of gradient updates
        total_losses = [loss / self._gradient_steps for loss in total_losses]

        return total_losses

    def _evaluate_policy(self, epoch, policy_hook=None):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        t0 = time()

        # Collect episodes for evaluation
        eval_trajectories, policy_hook_data = self._sampler.obtain_exact_episodes(
            n_eps_per_worker=self._num_evaluation_episodes,
            agent_update=self.get_updated_policy(policy_hook=policy_hook),
            env_updates=self.eval_env_updates,
        )

        # Log performance
        undiscounted_returns, log_dict = log_multitask_performance(
            epoch,
            batch=eval_trajectories,
            discount=self._discount,
            log_per_task=self._log_per_task
        )
        log_dict["average_return"] = np.mean(undiscounted_returns)

        logging.warn(f"Time to evaluate policy: {time()-t0:.2f}")

        return undiscounted_returns, log_dict, policy_hook_data

    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        log_dict = {}
        with torch.no_grad():
            log_dict["AlphaTemperature/mean"] = self._log_alpha.exp().mean().item()
        log_dict["Policy/Loss"] = policy_loss.item()
        log_dict["QF/{}".format("Qf1Loss")] = float(qf1_loss)
        log_dict["QF/{}".format("Qf2Loss")] = float(qf2_loss)
        log_dict["ReplayBuffer/buffer_size"] = self.replay_buffer.n_transitions_stored
        log_dict["Average/TrainAverageReturn"] = np.mean(self.episode_rewards)
        log_dict["TotalEnvSteps"] = self._total_envsteps

        return log_dict

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Raises:
            ValueError: If the number of tasks, num_tasks passed to
                this algorithm doesn't match the length of the task
                one-hot id in the observation vector.
        Returns:
            torch.Tensor: log_alpha. shape is (1, self.buffer_batch_size)
        """
        obs = samples_data["observation"]
        log_alpha = self._log_alpha
        one_hots = obs[:, -self._num_tasks:]

        if (log_alpha.shape[0] != one_hots.shape[1]
                or one_hots.shape[1] != self._num_tasks
                or log_alpha.shape[0] != self._num_tasks):
            raise ValueError(
                "The number of tasks in the environment does "
                "not match self._num_tasks. Are you sure that you passed "
                "The correct number of tasks?")

        with autocast(enabled=self._fp16):
            return torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()

    def _temperature_objective(self, log_pi, samples_data):
        """Compute the temperature/alpha coefficient loss.
        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.
        """
        alpha_loss = 0

        with autocast(enabled=self._fp16):
            if self._use_automatic_entropy_tuning:
                alpha_loss = (-(self._get_log_alpha(samples_data)) *
                            (log_pi.detach() + self._target_entropy)).mean()

            return alpha_loss

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        """Compute the Policy/Actor loss.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions (torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions (torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: loss from the Policy/Actor.
        """
        obs = samples_data["observation"]

        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        with autocast(enabled=self._fp16):
            min_q_new_actions = torch.min(self._qf1(obs, new_actions),
                                          self._qf2(obs, new_actions))

            policy_objective = ((alpha * log_pi_new_actions) -
                                min_q_new_actions.flatten()).mean()

            return policy_objective

    def _critic_objective(self, samples_data):
        """Compute the Q-function/critic loss.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.
        """
        obs = samples_data["observation"]
        actions = samples_data["action"]
        rewards = samples_data["reward"].flatten()
        terminals = samples_data["terminal"].flatten()
        next_obs = samples_data["next_observation"]

        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        with autocast(enabled=self._fp16):
            q1_pred = self._qf1(obs, actions)
            q2_pred = self._qf2(obs, actions)

            new_next_actions_dist = self.policy(next_obs)[0]
            new_next_actions_pre_tanh, new_next_actions = (
                new_next_actions_dist.rsample_with_pre_tanh_value())
            new_log_pi = new_next_actions_dist.log_prob(
                value=new_next_actions,
                pre_tanh_value=new_next_actions_pre_tanh
            )

            target_q_values = torch.min(
                self._target_qf1(next_obs, new_next_actions),
                self._target_qf2(next_obs, new_next_actions)
            ).flatten() - (alpha * new_log_pi)

            with torch.no_grad():
                q_target = rewards * self._reward_scale + (
                    1. - terminals) * self._discount * target_q_values

            qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
            qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

            return qf1_loss, qf2_loss

    def optimize_policy(self, samples_data):
        """Optimize the policy q_functions, and temperature coefficient. Rezero
        model weights (if applicable) after each optimizer step.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        if self._fp16:
            return self.optimize_policy_with_autocast(samples_data)

        obs = samples_data["observation"]
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self._qf1_optimizer.step()
        self._qf1.apply(rezero_weights)

        self._qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self._qf2_optimizer.step()
        self._qf2.apply(rezero_weights)

        action_dists = self.policy(obs)[0]
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)
        self._policy_optimizer.zero_grad()
        policy_loss.backward()

        self._policy_optimizer.step()
        self.policy.apply(rezero_weights)

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def optimize_policy_with_autocast(self, samples_data):
        """Optimize the policy q_functions, and temperature coefficient. Rezero
        model weights (if applicable) after each optimizer step.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data["observation"]

        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        self._gs_qf1.scale(qf1_loss).backward()
        self._gs_qf1.step(self._qf1_optimizer)
        self._gs_qf1.update()
        self._qf1.apply(rezero_weights)

        self._qf2_optimizer.zero_grad()
        self._gs_qf2.scale(qf2_loss).backward()
        self._gs_qf2.step(self._qf2_optimizer)
        self._gs_qf2.update()
        self._qf2.apply(rezero_weights)

        with autocast():
            action_dists = self.policy(obs)[0]
            new_actions_pre_tanh, new_actions = (
                action_dists.rsample_with_pre_tanh_value()
            )
            log_pi_new_actions = action_dists.log_prob(
                value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)

        self._policy_optimizer.zero_grad()
        self._gs_policy.scale(policy_loss).backward()
        self._gs_policy.step(self._policy_optimizer)
        self._gs_policy.update()
        self.policy.apply(rezero_weights)

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)

            self._alpha_optimizer.zero_grad()
            self._gs_alpha.scale(alpha_loss).backward()
            self._gs_alpha.step(self._alpha_optimizer)
            self._gs_alpha.update()

        return policy_loss, qf1_loss, qf2_loss

    def shutdown_worker(self):
        """Shutdown Plotter and Sampler workers."""
        self._sampler.shutdown_worker()