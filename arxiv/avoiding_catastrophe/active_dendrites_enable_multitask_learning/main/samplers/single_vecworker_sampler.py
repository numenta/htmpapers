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

from garage.experiment.deterministic import get_seed
from garage.sampler import FragmentWorker, LocalSampler
from garage.sampler.worker_factory import WorkerFactory

seed = get_seed()


class SingleVecWorkSampler(LocalSampler):
    """Sampler class which contains 1 vectorized worker which contains all the envs.

    The sampler need to be created either from a worker factory or from
    parameters which can construct a worker factory. See the __init__ method
    of WorkerFactory for the detail of these parameters.

    Args:
        agents (Policy or List[Policy]): Agent(s) to use to sample episodes.
            If a list is passed in, it must have length exactly
            `worker_factory.n_workers`, and will be spread across the
            workers.
        envs (Environment or List[Environment]): Environment from which
            episodes are sampled. If a list is passed in, it must have length
            exactly `worker_factory.n_workers`, and will be spread across the
            workers.
        worker_factory (WorkerFactory): Pickleable factory for creating
            workers. Should be transmitted to other processes / nodes where
            work needs to be done, then workers should be constructed
            there. Either this param or params after this are required to
            construct a sampler.
        max_episode_length(int): Params used to construct a worker factory.
            The maximum length episodes which will be sampled.
        is_tf_worker (bool): Whether it is workers for TFTrainer.
        seed(int): The seed to use to initialize random number generators.
        worker_class(type): Class of the workers. Instances should implement
            the Worker interface.
        worker_args (dict or None): Additional arguments that should be passed
            to the worker.

    """

    def __init__(
            self,
            agents,
            envs,
            *,  # After this require passing by keyword.
            worker_factory=None,
            max_episode_length=None,
            is_tf_worker=False,
            seed=seed,
            worker_class=FragmentWorker,
            worker_args=None):

        if worker_factory is None and max_episode_length is None:
            raise TypeError("Must construct a sampler from WorkerFactory or"
                            "parameters (at least max_episode_length)")
        if isinstance(worker_factory, WorkerFactory):
            self._factory = worker_factory
        else:
            self._factory = WorkerFactory(
                max_episode_length=max_episode_length,
                is_tf_worker=is_tf_worker,
                seed=seed,
                n_workers=1,
                worker_class=worker_class,
                worker_args=worker_args)

        self._agents = self._factory.prepare_worker_messages(agents)
        self._envs = [copy.deepcopy(env) for env in envs]
        self._num_envs = len(self._envs)
        self._workers = [
            self._factory(i) for i in range(self._factory.n_workers)
        ]
        self._workers[0].update_agent(self._agents[0])
        self._workers[0].update_env(self._envs)
        self.total_env_steps = 0

    def _update_workers(self, agent_update, env_update):
        """Apply updates to the workers.

        Args:
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        """
        agent_updates = self._factory.prepare_worker_messages(agent_update)
        # copy updates for all envs
        env_updates = [copy.deepcopy(env_update) for _ in range(self._num_envs)]
        # update the worker's agent, and all of the worker's envs
        self._workers[0].update_agent(agent_updates[0])
        self._workers[0].update_env(env_updates)
