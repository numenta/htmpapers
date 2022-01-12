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
import sys
import torch
import glob
import os

from args_parser import create_cmd_parser, create_exp_parser
from experiments import CONFIGS

from main.trainer import Trainer
from main.utils.parser_utils import merge_args


def find_latest_run(log_dir, exp_name):
    """
    Given a log dir and a projet name, return the latest project id
    """
    # Get full path to last experiment
    exp_path = os.path.join(log_dir, exp_name)
    last_exp = sorted(glob.glob(exp_path + "*"), key=os.path.getmtime, reverse=True)[0]
    # Return project id if last run is found, raise error otherwise
    if last_exp:
        project_id = os.path.basename(last_exp)[len(exp_name) + 1:]
        print(f"Restoring from last run found for this experiment: {project_id}")
        return project_id
    else:
        raise ValueError("No existing run could be found for this experiment,"
                         " cannot restore.")


if __name__ == "__main__":

    """
    Example usage:
        python run.py -e experiment_name
    """

    # Parse from command line
    cmd_parser = create_cmd_parser()
    run_args = cmd_parser.parse_args()
    if "exp_name" not in run_args:
        cmd_parser.print_help()
        exit(1)

    # Parse from config dictionary
    exp_parser = create_exp_parser()
    trainer_args = merge_args(exp_parser.parse_dict(CONFIGS[run_args.exp_name]))

    # Setup logging based on verbose defined by the user
    # TODO: logger level being overriden to WARN by garage, requires fixing
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if run_args.verbose else logging.INFO
    )
    logging.debug("Logger setup to debug mode")

    # Gives an additional option to define a wandb run name
    if run_args.wandb_run_name != "":
        run_args.wandb_run_name = run_args.exp_name

    # Restore and override project id from config with project id from cmd parser
    if run_args.restore:
        if not run_args.project_id:
            trainer_args.project_id = find_latest_run(
                trainer_args.log_dir, run_args.exp_name
            )
        else:
            trainer_args.project_id = run_args.project_id
            print(f"Overriding project id to {trainer_args.project_id}")

    # Automatically detects whether or not to use GPU
    use_gpu = torch.cuda.is_available() and not run_args.cpu
    print(f"Using GPU: {use_gpu}")
    print(trainer_args)

    trainer = Trainer(
        experiment_name=run_args.exp_name,
        use_gpu=use_gpu,
        trainer_args=trainer_args
    )

    if trainer_args.debug_mode:
        state_dict = trainer.state_dict()
    elif trainer_args.do_train:
        trainer.train(
            use_wandb=not run_args.local_only,
            evaluation_frequency=trainer_args.evaluation_frequency,
            checkpoint_frequency=trainer_args.checkpoint_frequency
        )