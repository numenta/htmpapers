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
Experiments with different approaches to setting workers for sampling
"""

from copy import deepcopy

from .base import base

# set to g4dn.8xlarge with 30 CPUs
# actually slower than using a single worker per env, up from 3.4sec to 4.5sec
multiple_workers_per_env = deepcopy(base)
multiple_workers_per_env = dict(
    num_tasks=10,
    workers_per_env=3,
    eval_episodes=3,
    cpus_per_worker=1,
    gpus_per_worker=0,
)

# down from 3.5 sec to 2.5sec on G instances
# and from 5 to about 4.5 on P instances
workers_on_gpu_only = deepcopy(base)
workers_on_gpu_only = dict(
    cpus_per_worker=0,
    gpus_per_worker=0.1,
)

# Export configurations in this file
CONFIGS = dict(
    multiworkers=multiple_workers_per_env,
    gpuworkers=workers_on_gpu_only,
)