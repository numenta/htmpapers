# -*- coding: utf-8 -*-
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
import argparse
import copy
import os
import re
from itertools import groupby
from pathlib import Path

import pandas as pd

from experiments import CONFIGS
from nupic.research.support import load_ray_tune_experiments


# Select a unique tag for each parameter combination, ignoring seed value
# Used to group multiple random seeds of the same configuration for computing results.
def key_func(x):
    s = re.split("[,]", re.sub(",|\\d+_|seed=\\d+", "", x["experiment_tag"]))
    if len(s[0]) == 0:
        return [" "]
    return s


def parse(best_result, results, trial_checkpoint, df_entries, exp, tag):
    config = trial_checkpoint["config"]
    model_args = config["model_args"]
    kw_percent_on = model_args["kw_percent_on"]
    weight_sparsity = model_args.get("weight_sparsity", 0.0)
    dendrite_weight_sparsity = model_args.get("dendrite_weight_sparsity", 0.0)
    num_segments = model_args.get("num_segments")
    dim_context = model_args["dim_context"]
    epochs = config["epochs"]
    num_tasks = config["num_tasks"]
    lr = config["optimizer_args"]["lr"]
    momentum = config["optimizer_args"].get("momentum", 0.0)
    iteration = results["training_iteration"]

    # This list must match the column headers in collect_results
    df_entries.append(
        [exp, kw_percent_on, weight_sparsity, dendrite_weight_sparsity, num_segments,
         dim_context, epochs, num_tasks, lr, momentum, config["seed"], best_result,
         iteration, "{} {}".format(exp, tag)]
    )


def parse_one_experiment(exp, state, df, outmethod):
    """
    Parse the trials in one experiment and append data to the given dataframe.

    :param exp: experiment name
    :param state: the `state` for the experiment. The state contains a list of runs.
                  Each run is an invocation from the command line. Each run can
                  consist of one or more trials.
    :param df: the dataframe to append to

    :return: a new dataframe with the results (the original one is not modified)
    """
    df_entries = []
    for experiment_state in state:
        # Go through all checkpoints in the experiment
        all_trials = experiment_state["checkpoints"]

        # Group trials based on their parameter combinations (represented by tag)
        parameter_groups = {
            k[0]: list(v)
            for k, v in groupby(sorted(all_trials, key=key_func), key=key_func)
        }

        for tag in parameter_groups:
            trial_checkpoints = parameter_groups[tag]

            try:
                for _, trial_checkpoint in enumerate(trial_checkpoints):
                    results = trial_checkpoint["results"]
                    if results is None:
                        continue
                    if outmethod == "best":
                        # For each checkpoint select the iteration with the best
                        # accuracy as the best epoch
                        print("using parsing method : best")
                        best_results = max(
                            results, key=lambda x: x.get("mean_accuracy", 0.0)
                        )
                        best_result = best_results["mean_accuracy"]
                        if best_result > 0.0:
                            parse(best_result, results, trial_checkpoint, df_entries,
                                  exp, tag)
                    elif outmethod == "lasttask":
                        print("using parsing method : lasttask")
                        last_results = results[-1]
                        last_result = last_results["mean_accuracy"]
                        if last_result > 0.0:
                            parse(last_result, last_results, trial_checkpoint,
                                  df_entries, exp, tag)
                    elif outmethod == "all":
                        print("using parsing method : all")
                        for i, _ in enumerate(results):
                            i_results = results[i]
                            i_result = i_results["mean_accuracy"]
                            if i_result > 0.0:
                                parse(i_result, i_results, trial_checkpoint,
                                      df_entries, exp, tag)

            except Exception:
                print(f"Problem with checkpoint group {tag} in {exp} ...skipping")
                continue

    # Create new dataframe from the entries with same dimensions as df
    df2 = pd.DataFrame(df_entries, columns=df.columns)
    return df.append(df2)


def collect_results(configs, basefilename, outmethod):
    """
    Parse the results for each specified experiment in each config file. Creates a
    dataframe containing one row for every trial for every network configuration in
    every experiment.

    The dataframe is saved to basefilename.pkl
    The raw results are also saved in a csv file named basefilename.csv.

    :param configs: list of experiment configs
    :param basefilename: base name for output files
    """

    # The results table
    columns = ["Experiment name", "Activation sparsity", "FF weight sparsity",
               "Dendrite weight sparsity", "Num segments", "Dim context", "Epochs",
               "Num tasks", "LR", "Momentum", "Seed", "Accuracy", "Iteration", "ID"]
    df = pd.DataFrame(columns=columns)

    for exp in configs:
        config = configs[exp]

        # Make sure path and data_dir are relative to the project location,
        # handling both ~/nta and ../results style paths.
        path = config.get("local_dir", ".")
        config["path"] = str(Path(path).expanduser().resolve())

        # Load experiment data
        experiment_path = os.path.join(config["path"], exp)
        try:
            states = load_ray_tune_experiments(
                experiment_path=experiment_path, load_results=True
            )

        except RuntimeError:
            print("Could not locate experiment state for " + exp + " ...skipping")
            continue

        df = parse_one_experiment(exp, states, df, outmethod)

    df.to_csv(f"{basefilename}_{outmethod}.csv")
    df.to_pickle(f"{basefilename}_{outmethod}.pkl")


def analyze_experiment_data(filename_df, output_filename):
    """
    Simple analysis to serve as an example. In general it's best to do the analysis in
    JupyterLab.

    :param filename_df: pickle filename containing the dataframe
    :param output_filename: filename to use to save the csv
    """
    df = pd.read_pickle(filename_df)
    df_id = df.groupby(["ID", "Seed"]).agg(
        num_trials=("ID", "count"),
        ff_weight_sparsity=("FF weight sparsity", "first"),
        activation_sparsity=("Activation sparsity", "first"),
        num_segments=("Num segments", "first"),
        mean_accuracy=("Accuracy", "mean"),
        stdev=("Accuracy", "std"),
    )
    # keep consistent columns names
    df_id.rename(
        columns={
            "num_trials": "ID",
            "ff_weight_sparsity": "FF weight sparsity",
            "activation_sparsity": "Activation sparsity",
            "num_segments": "Num segments",
            "mean_accuracy": "Accuracy",
            "stdev": "std",
        },
        inplace=True,
    )
    print(df_id)
    df_id.to_csv(output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+",
                        help="Experiments to run", choices=CONFIGS.keys())
    parser.add_argument("-f", dest="format", default="grid",
                        help="Table format", choices=["grid", "latex_raw"])
    parser.add_argument("-n", dest="name", default="temp", help="Base filename")
    parser.add_argument("-o", dest="outmethod", default="best",
                        help="Keep only considered task/run: best, last, or all")
    args = parser.parse_args()

    # Get configuration values
    configs = {}
    for name in args.experiments:
        configs[name] = copy.deepcopy(CONFIGS[name])

    collect_results(configs, args.name, args.outmethod)

    analyze_experiment_data(
        f"{args.name}_{args.outmethod}.pkl",
        f"{args.name}_{args.outmethod}_analysis.csv",
    )
