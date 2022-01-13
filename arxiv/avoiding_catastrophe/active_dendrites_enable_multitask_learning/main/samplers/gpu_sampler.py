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
import logging
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import ray
import torch
from garage import EpisodeBatch
from garage.experiment import deterministic
from garage.sampler import DefaultWorker, Sampler


class WorkerWithLogData():

    def rollout_eval(self, collect_hook_data=False):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: The collected episode.

        """
        self.start_episode()
        while not self.step_episode():
            pass
        eps_batch = self.collect_episode()

        hook_data = None
        if collect_hook_data:
            hook_data = self.agent.collect_hook_data()

        return eps_batch, hook_data


class WorkerWithEvalMode():

    def step_episode(self):
        """Take a single time-step in the current episode.

        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs)

            # Use the average for action selection when evaluating
            if self.eval_mode:
                a = agent_info["mean"]

            es = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if not es.terminal:
                self._prev_obs = es.observation
                return False
        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)
        return True


class CustomWorker(WorkerWithLogData, WorkerWithEvalMode, DefaultWorker):
    def __init__(
        self, seed, max_episode_length, worker_number, agent, env, device_type="cpu"
    ):
        self.deterministic_mode = True if seed else False
        self.device = torch.device(device_type)
        super().__init__(
            seed=seed,
            max_episode_length=max_episode_length,
            worker_number=worker_number
        )
        self.update(agent, env)

    def worker_init(self):
        """Initialize a worker."""
        if self._seed is not None and self.deterministic_mode:
            deterministic.set_seed(self._seed + self._worker_number)

    def update(self, agent_update, env_update, eval_mode=False):
        self.eval_mode = eval_mode
        self.update_agent(agent_update)
        self.update_env(env_update)

    def to_tensor(self, array):
        return torch.tensor(
            array,
            device=self.device,
            dtype=torch.float32
        )

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def start_episode(self):
        """Begin a new episode. Added moving observation to device"""
        self._eps_length = 0
        self._prev_obs, episode_info = self.env.reset()
        self._prev_obs = self.to_tensor(self._prev_obs)
        for k, v in episode_info.items():
            self._episode_infos[k].append(v)

        self.agent.reset()

    def step_episode(self):
        """Take a single time-step in the current episode.
        Added moving action to GPU

        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            es = self.env.step(a)
            self._observations.append(self.to_numpy(self._prev_obs))  # fix it
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if not es.terminal:
                self._prev_obs = self.to_tensor(es.observation)
                return False

        self._lengths.append(self._eps_length)
        self._last_observations.append(self.to_numpy(self._prev_obs))  # fix it
        return True

@ray.remote
class RayWorker(CustomWorker):
    pass


class RaySampler(Sampler):
    """Samples episodes in a data-parallel fashion using a Ray cluster.

    The sampler need to be created either from a worker factory or from
    parameters which can construct a worker factory. See the __init__ method
    of WorkerFactory for the detail of these parameters.

    Args:
        agents (list[Policy]): Agents to distribute across workers.
        envs (list[Environment]): Environments to distribute across workers.
        max_episode_length(int): Params used to construct a worker factory.
            The maximum length episodes which will be sampled.
        TODO: add other arguments
    """

    def __init__(
        self,
        agent,  # it is actually a single network
        envs,
        max_episode_length=None,
        seed=None,
        cpus_per_worker=1,
        gpus_per_worker=0,
        workers_per_env=1
    ):

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(log_to_driver=False, ignore_reinit_error=True)

        # Verifies max episode length
        if max_episode_length is None:
            raise TypeError("Requires max_episode_length to be defined")
        self.max_episode_length = max_episode_length

        assert (cpus_per_worker + gpus_per_worker > 0), \
            "Must allocate > 0 CPUs or GPUs to each worker"

        # Use either only GPU or CPU for a single worker
        cpus_per_worker = cpus_per_worker if gpus_per_worker <= 0 else 0

        # Create one worker per env
        self.workers_per_env = workers_per_env
        self.workers = []
        for _ in range(self.workers_per_env):
            self.workers.extend(
                [RayWorker.options(
                    # arguments for Ray
                    num_cpus=cpus_per_worker, num_gpus=gpus_per_worker
                ).remote(
                    # arguments for the worker class
                    seed=seed,
                    max_episode_length=max_episode_length,
                    worker_number=idx,
                    agent=agent,
                    env=env,
                    device_type="cuda" if gpus_per_worker > 0 else "cpu"
                ) for idx, env in enumerate(envs)]
            )

        logging.warn(
            f"Creating {len(self.workers)} workers with {cpus_per_worker:.2f} cpus "
            f"and {gpus_per_worker:.2f} gpus"
        )

        # Extra attributes
        self.total_env_steps = 0
        self.device = torch.device("cuda" if gpus_per_worker > 0 else "cpu")

    def update_workers_eval(self, agent_update, env_updates, eval_mode=False):
        """Update all of the workers

        Args:
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling_episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
        """
        agent_update.to(self.device)
        agent_update.eval()
        for p in agent_update.parameters():
            p.requires_grad_(False)

        pids = [
            worker.update.remote(deepcopy(agent_update), env_update, eval_mode)
            for env_update, worker in zip(env_updates, self.workers)
        ]
        for pid in pids:
            ray.get(pid)

    def update_workers(self, agent_update, env_updates, eval_mode=False):
        """Update all of the workers

        Args:
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_updates (object): Value which will be passed into the
                `env_update_fn` before sampling_episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
        """

        # Set to eval and turn off all gradient updates for inference only network
        agent_update = agent_update.to(self.device)
        agent_update.eval()
        for p in agent_update.parameters():
            p.requires_grad_(False)

        assert len(env_updates) == len(self.workers), \
            "Number of envs should match number of workers"

        pids = [
            worker.update.remote(agent_update, env_update, eval_mode)
            for env_update, worker in zip(env_updates, self.workers)
        ]
        for pid in pids:
            ray.get(pid)


    def obtain_samples(self, num_samples, agent_update, env_updates=None):
        """Sample the policy for new episodes.

        Args:
            num_samples (int): Number of steps the the sampler should collect.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_updates (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: Batch of gathered episodes.

        """

        self.update_workers(agent_update, env_updates)
        completed_samples = 0
        batches = []

        # TODO: can we replace the while, so all processes are scheduled beforehand?
        while completed_samples < num_samples:
            pids = [w.rollout.remote() for w in self.workers]
            results = [ray.get(pid) for pid in pids]
            for episode_batch in results:
                num_returned_samples = episode_batch.lengths.sum()
                completed_samples += num_returned_samples
                batches.append(episode_batch)

        # Note: EpisodeBatch takes care of concatenating - is this a performance issue?
        samples = EpisodeBatch.concatenate(*batches)
        self.total_env_steps += sum(samples.lengths)
        return samples

    def obtain_exact_episodes(
        self,
        n_eps_per_worker,
        agent_update,
        collect_hook_data=False,
        env_updates=None
    ):
        """Sample an exact number of episodes per worker.

        Args:
            n_eps_per_worker (int): Exact number of episodes to gather for
                each worker.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: Batch of gathered episodes. Always in worker
                order. In other words, first all episodes from worker 0, then
                all episodes from worker 1, etc.

        """

        self.update_workers_eval(agent_update, env_updates, eval_mode=True)

        # adjust n_eps_per_worker to account for the number of workers available per env
        assert n_eps_per_worker % self.workers_per_env == 0, \
            "Number of eps per worker should be a multiple of workers per env"
        n_eps_per_worker = int(n_eps_per_worker / self.workers_per_env)

        # only include logdata if hook has been attached.
        if(hasattr(agent_update, "collect_hook_data")):
            collect_hook_data = True

        episodes = defaultdict(list)
        data_to_export = defaultdict(dict)

        def update_eval_results(results):
            for worker_id, (episode_batch, hook_data) in enumerate(results):
                episodes[worker_id].append(episode_batch)
                if hook_data is not None:
                    # Loop through hooks, allow multiple hooks
                    for hook, data in hook_data.items():
                        # For each hook save data for each task separately
                        # allowing for several episodes to be saved at once
                        if worker_id not in data_to_export[hook]:
                            data_to_export[hook][worker_id] = data
                        else:
                            data_to_export[hook][worker_id] = torch.cat(
                                [data_to_export[hook][worker_id], data], dim=0
                            )

        # TODO: do it all async, including loop through episodes if more than one
        for _ in range(n_eps_per_worker):
            pids = [
                worker.rollout_eval.remote(collect_hook_data=collect_hook_data)
                for worker in self.workers
            ]
            episode_results = [ray.get(pid) for pid in pids]
            update_eval_results(episode_results)

        # Note: do they need to be ordered?
        ordered_episodes = list(chain(
            *[episodes[i] for i in range(len(self.workers))]
        ))

        samples = EpisodeBatch.concatenate(*ordered_episodes)  # concat
        self.total_env_steps += sum(samples.lengths)
        return samples, data_to_export

    def shutdown_worker(self):
        """Shuts down all workers and Ray"""
        # close environments prior to terminating worker
        pids = [w.shutdown.remote() for w in self.workers]
        for pid in pids:
            ray.get(pid)
        # kill Ray Actors and shutdown Ray
        for worker in self.workers:
            ray.kill(worker)
        ray.shutdown()


class RaySamplerSyncEval(RaySampler):
    """
    Copy of RaySampler, but with synchronous evaluation.
    Used for debugging purposes only.
    """

    def __init__(
        self,
        agent,
        envs,
        max_episode_length=None,
        seed=None,
        cpus_per_worker=1,
        gpus_per_worker=0,
        workers_per_env=1
    ):
        super().__init__(
            agent,
            envs,
            max_episode_length,
            seed,
            cpus_per_worker,
            gpus_per_worker,
            workers_per_env
        )
        # non Ray workers for evaluation
        self.eval_workers = []
        for _ in range(self.workers_per_env):
            self.eval_workers.extend(
                [CustomWorker(
                    seed=seed,
                    max_episode_length=max_episode_length,
                    worker_number=idx,
                    agent=agent,
                    env=env,
                    device_type="cuda" if gpus_per_worker > 0 else "cpu"
                ) for idx, env in enumerate(envs)]
            )

    def update_workers_eval(self, agent_update, env_updates, eval_mode=False):
        """Update all of the workers

        Args:
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling_episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
        """
        if env_updates is None:
            env_updates = [None] * len(self.eval_workers)

        agent_update.to(self.device)
        agent_update.eval()
        for p in agent_update.parameters():
            p.requires_grad_(False)

        # sync version
        for env_update, worker in zip(env_updates, self.eval_workers):
            worker.update(deepcopy(agent_update), env_update, eval_mode)

    def obtain_exact_episodes(
        self,
        n_eps_per_worker,
        agent_update,
        collect_hook_data=False,
        env_updates=None
    ):
        """Sample an exact number of episodes per worker.

        Args:
            n_eps_per_worker (int): Exact number of episodes to gather for
                each worker.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: Batch of gathered episodes. Always in worker
                order. In other words, first all episodes from worker 0, then
                all episodes from worker 1, etc.

        """

        self.update_workers_eval(agent_update, env_updates, eval_mode=True)

        # adjust n_eps_per_worker to account for the number of workers available per env
        assert n_eps_per_worker % self.workers_per_env == 0, \
            "Number of eps per worker should be a multiple of workers per env"
        n_eps_per_worker = int(n_eps_per_worker / self.workers_per_env)

        # only include hook data if hook has been attached.
        if(hasattr(agent_update, "collect_hook_data")):
            collect_hook_data = True

        episodes = defaultdict(list)
        data_to_export = defaultdict(dict)

        def update_eval_results(results):
            for worker_id, (episode_batch, hook_data) in enumerate(results):
                episodes[worker_id].append(episode_batch)
                if hook_data is not None:
                    # Loop through hooks, allow multiple hooks
                    for hook, data in hook_data.items():
                        # For each hook save data for each task separately
                        # allowing for several episodes to be saved at once
                        if worker_id not in data_to_export[hook]:
                            data_to_export[hook][worker_id] = data
                        else:
                            data_to_export[hook][worker_id] = torch.cat(
                                [data_to_export[hook][worker_id], data], dim=0
                            )

        for _ in range(n_eps_per_worker):
            episode_results = [
                worker.rollout_eval(collect_hook_data=collect_hook_data)
                for worker in self.eval_workers
            ]
            update_eval_results(episode_results)

        # Note: do they need to be ordered?
        ordered_episodes = list(chain(
            *[episodes[i] for i in range(len(self.eval_workers))]
        ))

        samples = EpisodeBatch.concatenate(*ordered_episodes)  # concat
        self.total_env_steps += sum(samples.lengths)
        return samples, data_to_export


class SamplerEvalOnly(RaySamplerSyncEval):
    """
    Sampler version that only does eval on cpu
    Overrides init to remove initialization of Ray and trainer workers.
    Uses SyncEval methods from parent class
    """

    def __init__(
        self,
        agent,
        envs,
        max_episode_length=None,
        seed=None,
        cpus_per_worker=1,
        gpus_per_worker=0,
        workers_per_env=1
    ):

        self.workers_per_env = workers_per_env
        self.total_env_steps = 0
        self.device = torch.device("cuda" if gpus_per_worker > 0 else "cpu")

        # non Ray workers for evaluation
        self.eval_workers = []
        for _ in range(self.workers_per_env):
            self.eval_workers.extend(
                [CustomWorker(
                    seed=seed,
                    max_episode_length=max_episode_length,
                    worker_number=idx,
                    agent=agent,
                    env=env,
                    device_type="cuda" if gpus_per_worker > 0 else "cpu"
                ) for idx, env in enumerate(envs)]
            )
