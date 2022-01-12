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
import logging
import os
import cloudpickle
from threading import Thread

import metaworld
import numpy as np
import torch
import wandb
from garage.experiment import deterministic
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.torch import set_gpu_mode

from main.algorithms.custom_mtsac import CustomMTSAC
from main.samplers.gpu_sampler import RaySampler
from main.utils.garage_utils import (
    calculate_mean_param,
    create_policy_net,
    create_qf_net,
)
from main.utils.parser_utils import dict_to_dataclass


class Trainer():
    """Custom trainer class which
    - removes Tabular usage
    - refactors bidirectional message in run epoch intro traditional OOP architecture
    """

    def __init__(
        self,
        experiment_name,
        use_gpu,
        trainer_args
    ):
        """Train MTSAC with metaworld_experiments environment.
        Args:
            experiment_name: expeirment name to be used for logging and checkpointing
            use_wandb: boolean, defines whether or not to log to wandb
            use_gpu: boolean, defines whether or not to use to GPU for training
            trainer_args: named tuple with args given by config
        """

        # Define log and checkpoint dir
        self.checkpoint_dir = os.path.join(
            trainer_args.log_dir,
            f"{experiment_name}-{trainer_args.project_id}"
        )
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        self.state_path = os.path.join(self.checkpoint_dir, "experiment_state.p")
        self.env_state_path = os.path.join(self.checkpoint_dir, "env_state.p")
        self.config_path = os.path.join(self.checkpoint_dir, "config.json")
        self.experiment_name = experiment_name

        # Only define viz_save_path if required to save visualizations local
        self.viz_save_path = None
        if trainer_args.save_visualizations_local:
            self.viz_save_path = os.path.join(self.checkpoint_dir, "viz")

        # Check if loading from existing experiment
        self.loading_from_existing = os.path.exists(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save arguments for later retrieval
        self.init_config(trainer_args)

        num_tasks = trainer_args.num_tasks

        # TODO: do we have to fix which GPU to use? run distributed across multiGPUs
        if use_gpu:
            set_gpu_mode(True, 0)

        if trainer_args.seed is not None:
            deterministic.set_seed(trainer_args.seed)

        # Note: different classes whether it uses 10 or 50 tasks. Why?
        mt_env = (
            metaworld.MT10(seed=trainer_args.env_seed) if num_tasks <= 10
            else metaworld.MT50(seed=trainer_args.env_seed)
        )

        train_task_sampler = MetaWorldTaskSampler(
            mt_env, "train", add_env_onehot=True
        )

        # TODO: add some clarifying comments of why these asserts are required
        assert num_tasks % 10 == 0, "Number of tasks have to divisible by 10"
        assert num_tasks <= 500, "Number of tasks should be less or equal 500"

        # TODO: do we have guarantees that in case seed is set, the tasks being sampled
        # are the same?
        mt_train_envs = train_task_sampler.sample(num_tasks)
        env = mt_train_envs[0]()

        if trainer_args.params_seed is not None:
            torch.manual_seed(trainer_args.params_seed)

        policy = create_policy_net(env_spec=env.spec, net_params=trainer_args)
        qf1 = create_qf_net(env_spec=env.spec, net_params=trainer_args)
        qf2 = create_qf_net(env_spec=env.spec, net_params=trainer_args)

        if trainer_args.params_seed is not None:
            calculate_mean_param("policy", policy)
            calculate_mean_param("qf1", qf1)
            calculate_mean_param("qf2", qf2)

        if trainer_args.override_weight_initialization:
            logging.warn("Overriding dendritic layer weight initialization")
            self.override_weight_initialization([policy, qf1, qf2])

        replay_buffer = PathBuffer(
            capacity_in_transitions=trainer_args.num_buffer_transitions
        )
        max_episode_length = env.spec.max_episode_length
        self.env_steps_per_epoch = int(max_episode_length * num_tasks)
        self.num_epochs = trainer_args.timesteps // self.env_steps_per_epoch

        sampler = RaySampler(
            agent=policy,
            envs=mt_train_envs,
            max_episode_length=max_episode_length,
            cpus_per_worker=trainer_args.cpus_per_worker,
            gpus_per_worker=trainer_args.gpus_per_worker,
            workers_per_env=trainer_args.workers_per_env,
            seed=trainer_args.seed,
        )

        self._algo = CustomMTSAC(
            env_spec=env.spec,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            sampler=sampler,
            train_task_sampler=train_task_sampler,
            gradient_steps_per_itr=int(
                max_episode_length * trainer_args.num_grad_steps_scale
            ),
            task_update_frequency=trainer_args.task_update_frequency,
            num_tasks=num_tasks,
            min_buffer_size=max_episode_length * num_tasks,
            target_update_tau=trainer_args.target_update_tau,
            discount=trainer_args.discount,
            buffer_batch_size=trainer_args.buffer_batch_size,
            policy_lr=trainer_args.policy_lr,
            qf_lr=trainer_args.qf_lr,
            reward_scale=trainer_args.reward_scale,
            num_evaluation_episodes=trainer_args.eval_episodes,
            fp16=trainer_args.fp16 if use_gpu else False,
            log_per_task=trainer_args.log_per_task,
            share_train_eval_env=trainer_args.share_train_eval_env
        )

        # Override with loaded networks if existing experiment
        self.current_epoch = 0
        if self.loading_from_existing:
            self.load_experiment_state()

        # Move all networks within the model on device
        self._algo.to()

    def override_weight_initialization(self, networks):
        """Override weight initialization for dendrite layers"""
        if type(networks) != list:
            networks = [networks]

        for network in networks:
            for name, layer in network.named_modules():
                if isinstance(layer, torch.nn.Sequential) and "dendrite" in name:
                    segments = layer[0].segments
                    linear_dim, segments_dim, context_dim = segments.weights.shape
                    new_weights = []
                    for _ in range(segments_dim):
                        new_weights.append(
                            (torch.rand(linear_dim, context_dim).unsqueeze(dim=1) - 0.5)
                            / np.sqrt(linear_dim + context_dim)
                        )
                    new_weights = torch.cat(new_weights, dim=1)
                    segments.weights.data = new_weights

    def save_experiment_state(self):
        print("***Saving experiment state")

        def save_fn(state_dict, state_path, env_state, env_state_path):
            torch.save(state_dict, self.state_path)
            with open(env_state_path, "wb") as f:
                cloudpickle.dump(env_state, f)

        # save state dict
        thread = Thread(target=lambda: save_fn(
            self.state_dict(),
            self.state_path,
            self._algo.eval_env_updates,
            self.env_state_path
        ))
        thread.start()
        thread.join()

    def load_experiment_state(self):
        print(f"***Loading experiment state from {self.state_path}")
        experiment_state = torch.load(self.state_path)
        self._algo.load_state(experiment_state["algorithm"])

        with open(self.env_state_path, "rb") as f:
            env_state = cloudpickle.load(f)
        self._algo.load_env_state(env_state)

        self.current_epoch = experiment_state["current_epoch"]

        if self.current_epoch == self.num_epochs:
            logging.warn("Loading from existing experiment that has already finished.")

    def state_dict(self):
        return {
            "algorithm": self._algo.state_dict(),
            "current_epoch": self.current_epoch
        }

    def init_config(self, trainer_args):
        """Load if existing. Gives warning if configs doesn't match"""
        if self.loading_from_existing:
            with open(self.config_path, "r") as file:
                args_loaded = json.load(file)
                updated_args = self.merge_loaded(trainer_args, args_loaded)
                self.trainer_args = dict_to_dataclass(updated_args)
        else:
            self.trainer_args = trainer_args
            with open(self.config_path, "w") as file:
                export_dict = {
                    k: v for k, v in self.trainer_args.__dict__.items()
                    if not callable(v)
                }
                json.dump(export_dict, file)

    def merge_loaded(self, original, loaded):
        """
        Merge config from loaded checkpoing with config defined by experiment name.
        Required since not callables are not saved to json.
        Report if any inconsistencies are found.
        """
        updated_args = {}
        for key, original_value in original.__dict__.items():
            # If key in loaded config, use value from loaded config
            if key in loaded:
                new_value = loaded[key]
                updated_args[key] = new_value
                if original_value != new_value:
                    logging.warn(
                        f"For key {key}, value defined in config {original_value}"
                        f" doesn't equal value saved in checkpoint {new_value}"
                    )
            # If key not in loaded config, use original one
            else:
                updated_args[key] = original_value
        return updated_args

    def train(
        self,
        use_wandb=True,
        evaluation_frequency=25,
        checkpoint_frequency=25,
    ):
        """
        Start training.
        This version replaces step_epochs in original garage Trainer

        Args:
            num_epochs (int): Number of epochs.
            batch_size (int or None):
                Number of environment steps (samples) collected per epoch
                Only defines how many samples will be collected and added to buffer
                Does not define size of the batch used in the gradient update.

            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.

        Raises:
            NotSetupError: If train() is called before setup().

        Returns:
            float: The average return in last epoch cycle.

        """
        # Log experiment json file

        logging.info("Starting training...")
        if use_wandb:
            wandb.init(
                name=self.experiment_name,
                project=self.trainer_args.project_name,
                group=self.trainer_args.wandb_group,
                reinit=True,
                config=self.trainer_args.__dict__,
                id=self.trainer_args.project_id,
                dir=self.checkpoint_dir,
                resume="allow" if self.loading_from_existing else None,
            )

        # Loop through epochs and call algorithm to run one epoch at at ime
        for epoch in range(self.current_epoch, self.num_epochs):

            # Run training epoch
            log_dict = self._algo.run_epoch(
                epoch=epoch, env_steps_per_epoch=self.env_steps_per_epoch
            )

        # Run evaluation, with a given frequency
            if epoch % evaluation_frequency == 0:
                # Evalutes with optional hook to collect data
                hook_manager_class = self.trainer_args.policy_data_collection_hook
                eval_returns, eval_log_dict, hook_data = self._algo._evaluate_policy(
                    epoch, hook_manager_class
                )
                log_dict["average_return"] = np.mean(eval_returns)
                log_dict.update(eval_log_dict)
                # Reports data from hook
                if hook_manager_class is not None:
                    hook_log_dict = hook_manager_class.consolidate_and_report(
                        hook_data,
                        epoch=epoch,
                        local_save_path=self.viz_save_path
                    )
                    log_dict.update(hook_log_dict)

            self.current_epoch = epoch + 1
            log_dict["epoch"] = self.current_epoch

            # Logging and updating state variables
            if use_wandb:
                wandb.log(log_dict)

            # Checkpoint with given frequency
            if checkpoint_frequency is not None and epoch % checkpoint_frequency == 0:
                self.save_experiment_state()

        self._algo.shutdown_worker()
