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

import argparse
import logging
import numpy as np
import os
from dataclasses import dataclass, field
from experiments import CONFIGS
from typing import Optional, Tuple, Callable
from typing_extensions import Literal
from main.utils.parser_utils import DataClassArgumentParser, create_id

@dataclass
class LoggingArguments:
    project_name: str = "multitask"
    log_dir: Optional[str] = None
    project_id: str = create_id()
    wandb_group: Optional[str] = None
    log_per_task: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to log individual results per task."
        }
    )
    policy_data_collection_hook: Optional[Callable] = None
    save_visualizations_local: bool = True

    def __post_init__(self):
        if self.log_dir is None:
            logging.warn(
                "log_dir is not defined, attempting to default "
                "to environment variable CHECKPOINT_DIR/multitask"
            )
            if "CHECKPOINT_DIR" not in os.environ:
                raise KeyError(
                    "Environment variable CHECKPOINT_DIR not found, required "
                    "when log_dir is not defined in experiment config"
                )
            else:
                self.log_dir = os.path.join(
                    os.environ["CHECKPOINT_DIR"],
                    "embodiedai",
                    "multitask"
                )
                logging.warn(f"Defining log_dir as {self.log_dir}")


@dataclass
class ExperimentArguments:
    seed: Optional[int] = None
    env_seed: Optional[int] = np.random.randint(10e4)
    params_seed: Optional[int] = None
    timesteps: int = 15000000
    cpus_per_worker: float = 0.5
    gpus_per_worker: float = 0
    workers_per_env: int = 1
    do_train: bool = True
    debug_mode: bool = False
    use_deterministic_evaluation: bool = False
    fp16: bool = False
    checkpoint_frequency: int = 100
    override_weight_initialization: bool = False

@dataclass
class TrainingArguments:
    discount: float = 0.99
    eval_episodes: int = 3
    num_buffer_transitions: float = 1e6
    evaluation_frequency: int = 10
    task_update_frequency: int = 1
    share_train_eval_env: bool = False
    target_update_tau: float = 5e-3
    buffer_batch_size: int = 2560
    num_grad_steps_scale: float = 0.5
    policy_lr: float = 3.91e-4
    qf_lr: float = 3.91e-4
    reward_scale: float = 1.0

    def __post_init__(self):
        self.num_buffer_transitions = int(self.num_buffer_transitions)


@dataclass
class NetworkArguments:
    net_type: str = "Dendrite_MLP"
    num_tasks: int = 10
    input_data: Literal["obs", "obs|context"] = field(
        default="obs",
        metadata={
            "help": "(str): Type of input data to use. Can be either 'obs|context'"
                    "(observation concatenated with context, which is a one-hot"
                    "encoded task id) or 'obs' (just the obseration)."
        }
    )
    context_data: Literal["context", "obs|context", None] = field(
        default="obs|context",
        metadata={
            "help": "(str) Type of context data to use. Can be either 'obs|context'"
                    "(observation concatenated with context, which is a one-hot"
                    "encoded task id) or 'context' (just the one-hot encoded task id)."
        }
    )
    hidden_sizes: Tuple = (2048, 2048)
    layers_modulated: Tuple = tuple(range(len(hidden_sizes)))
    num_segments: int = 1
    kw_percent_on: float = 0.33
    context_percent_on: float = 1.0
    weight_sparsity: float = 0.0
    weight_init: str = "modified"
    dendrite_weight_sparsity: float = 0.0
    dendrite_init: str = "modified"
    dendritic_layer_class: str = "one_segment"
    output_nonlinearity: Optional[str] = None
    preprocess_module_type: str = "relu"
    preprocess_output_dim: int = 64
    policy_min_log_std: float = -20.0
    policy_max_log_std: float = 2.0
    policy_hidden_nonlinearity: str = "relu"
    qf_hidden_nonlinearity: str = "relu"
    distribution: str = "TanhNormal"

    def __post_init__(self):
        # TODO: modify DataClassArgumentParser to verify if Literals are valid

        assert self.input_data in {"obs", "obs|context"}
        assert self.context_data in {"context", "obs|context", None}
        assert self.kw_percent_on is None or (self.kw_percent_on >= 0.0 and self.kw_percent_on < 1.0)
        assert self.context_percent_on >= 0.0
        assert self.weight_init in {"modified", "kaiming"}
        assert self.dendrite_init in {"modified", "kaiming"}
        assert self.preprocess_module_type in {None, "relu", "kw"}

        if self.net_type == "Dendrite_MLP":
            assert self.num_segments >= 1

            if self.num_segments == 1 or self.dendritic_layer_class == "one_segment":
                self.num_segments = 1
                self.dendritic_layer_class = "one_segment"
            elif self.num_segments > 1:
                assert self.dendritic_layer_class in {"biasing", "max_gating", "abs_max_gating_signed", "abs_max_gating_unsigned"}
        elif self.net_type == "MLP":
            self.input_data = "obs|context"
            self.context_data = None
            self.num_segments = 0
            self.dendritic_layer_class = None


        if self.kw_percent_on == 0.0:
            self.kw_percent_on = None


def create_exp_parser():
    return DataClassArgumentParser([
        LoggingArguments,
        ExperimentArguments,
        TrainingArguments,
        NetworkArguments,
    ])

def create_cmd_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        add_help=False,
    )
    # Run options
    parser.add_argument(
        "-e",
        "--exp_name",
        help="Experiment to run",
        choices=list(CONFIGS.keys())
    )
    parser.add_argument(
        "-n",
        "--wandb_run_name",
        default="",
        help="Name of the wandb run"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--local_only",
        action="store_true",
        default=False,
        help="Whether or not to log results to wandb"
    )
    parser.add_argument(
        "-c",
        "--cpu",
        action="store_true",
        default=False,
        help="Whether to use CPU even if GPU is available",
    )
    parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        default=False,
        help="Whether to restore from existing experiment with same project name",
    )
    parser.add_argument(
        "-p",
        "--project_id",
        default=None,
        help="Alternative way of providing project id",
    )

    # TODO: evaluate whether or not debugging flag is required
    parser.add_argument(
        "-d",
        "--debugging",
        action="store_true",
        default=False,
        help="Option to use when debugging so not every test run is logged.",
    )

    return parser
