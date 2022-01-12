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

"""
This module trains & evaluates a dendritic network in a continual learning setting on
permutedMNIST for a specified number of tasks/permutations. A context vector is
provided to the dendritic network, so task information need not be inferred.

This setup is very similar to that of context-dependent gating model from the paper
'Alleviating catastrophic forgetting using contextdependent gating and synaptic
stabilization' (Masse et al., 2018).
"""
import argparse
import copy

from experiments import CONFIGS
from nupic.research.frameworks.ray.run_with_raytune import run as run_with_ray_tune
from nupic.research.frameworks.vernon.parser_utils import (
    get_default_parsers,
    process_args,
)
from nupic.research.frameworks.vernon.run import run as run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=get_default_parsers(),
    )

    parser.add_argument("-e", "--experiment", dest="name", nargs="+",
                        default="default_base", help="Experiment to run",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--run_without_ray_tune", dest="run_without_ray_tune",
                        type=bool, default=False,
                        help="run by calling ray.run_with_ray_tune or vernon.run")

    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
        exit(1)

    for experiment in args.name:

        # Get configuration values
        config = copy.deepcopy(CONFIGS[experiment])

        # Merge configuration with command line arguments
        config.update(vars(args))
        config.update(name=experiment)

        config = process_args(args, config)

        if config is None:
            continue
        if args.run_without_ray_tune:
            run(config)
        else:
            run_with_ray_tune(config)
