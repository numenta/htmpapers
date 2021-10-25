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
# ----------------------------------------------------------------------
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt
import seaborn as sns

sns.set(style="ticks", font_scale=1.3)


def hyperparameter_dendrites_wit_si_search_panel():
    """
    Plots a 3 panels figure on 1 rows x 3 columns
    Rows contains figures representing hyperparameters search for 10
    permutedMNIST tasks resulting from hyperparameter_search.py config file.
    Columns 1 is the number of dendritic segments, columns 2 the activation
    sparsity and column 3 the weight sparsity.
    """

    df_path1 = f"{experiment_folder}si_centroid_hp_10_lasttask.csv"
    df1 = pd.read_csv(df_path1)

    df_path2 = f"{experiment_folder}si_centroid_hp_10_control_lasttask.csv"
    df2 = pd.read_csv(df_path2)

    df1 = df1[["Activation sparsity", "FF weight sparsity", "Num segments",
               "Accuracy"]]
    df2 = df2[["Activation sparsity", "FF weight sparsity", "Num segments",
               "Accuracy"]]
    df1["condition"] = "dendrite_and_ff"
    df2["condition"] = "ff_only"
    df = pd.concat([df1, df2])

    gs = gridspec.GridSpec(1, 1)
    fig = plt.figure(figsize=(5, 5))

    ax1 = fig.add_subplot(gs[0, 0])

    x1 = "Num segments"
    dhue = "condition"
    y = "Accuracy"
    ort = "v"
    pal = sns.color_palette(n_colors=9)
    sigma = 0.2
    fig.suptitle(
        "Impact of the number of segments with SI on performance", fontsize=12
    )

    pt.RainCloud(x=x1, y=y, hue=dhue, data=df, palette=pal, bw=sigma,
                 width_viol=0.6, ax=ax1, orient=ort, move=0.2, pointplot=True,
                 alpha=0.65)

    ax1.set_ylabel("Mean accuracy", fontsize=16)
    ax1.set_xlabel("Number of dendritic segments", fontsize=16)

    if savefigs:
        plt.savefig(
            f"{figs_dir}/hyperparameter_search_panel_with_si_dendrites.png",
            bbox_inches="tight"
        )


if __name__ == "__main__":

    savefigs = True
    figs_dir = "figs/"
    if savefigs:
        if not os.path.isdir(f"{figs_dir}"):
            os.makedirs(f"{figs_dir}")

    experiment_folder = "data_hyperparameter_search_with_si_dendrites/"

    hyperparameter_dendrites_wit_si_search_panel()
