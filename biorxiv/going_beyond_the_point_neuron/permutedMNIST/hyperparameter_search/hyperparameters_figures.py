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
import matplotlib.ticker as ticker
import pandas as pd
import ptitprince as pt
import seaborn as sns

sns.set(style="ticks", font_scale=1.3)


def hyperparameter_search_panel():
    """
    Plots a 6 panels figure on 2 rows x 3 columns
    Rows contains figures representing hyperparameters search for 10 and 50
    permutedMNIST tasks resulting from hyperparameter_search.py config file.
    Columns 1 is the number of dendritic segments, columns 2 the activation
    sparsity and column 3 the weight sparsity.
    """

    df_path1 = f"{experiment_folder}segment_search_lasttask.csv"
    df1 = pd.read_csv(df_path1)

    df_path2 = f"{experiment_folder}kw_sparsity_search_lasttask.csv"
    df2 = pd.read_csv(df_path2)

    df_path3 = f"{experiment_folder}w_sparsity_search_lasttask.csv"
    df3 = pd.read_csv(df_path3)

    df_path1_50 = f"{experiment_folder}segment_search_50_lasttask.csv"
    df1_50 = pd.read_csv(df_path1_50)

    df_path2_50 = f"{experiment_folder}kw_sparsity_search_50_lasttask.csv"
    df2_50 = pd.read_csv(df_path2_50)

    df_path3_50 = f"{experiment_folder}w_sparsity_search_50_lasttask.csv"
    df3_50 = pd.read_csv(df_path3_50)

    relevant_columns = ["Activation sparsity", "FF weight sparsity", "Num segments",
                        "Accuracy"]

    df1 = df1[relevant_columns]
    df2 = df2[relevant_columns]
    df3 = df3[relevant_columns]
    df1_50 = df1_50[relevant_columns]
    df2_50 = df2_50[relevant_columns]
    df3_50 = df3_50[relevant_columns]

    # Figure 1 'Impact of the different hyperparameters on performance
    # full cross product of hyperparameters
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax1_50 = fig.add_subplot(gs[1, 0])
    ax2_50 = fig.add_subplot(gs[1, 1])
    ax3_50 = fig.add_subplot(gs[1, 2])

    x1 = "Num segments"
    x2 = "Activation sparsity"
    x3 = "FF weight sparsity"

    y = "Accuracy"
    ort = "v"
    pal = sns.color_palette(n_colors=6)
    sigma = 0.2
    fig.suptitle(
        "Impact of the different hyperparameters on performance", fontsize=12
    )

    pt.RainCloud(x=x1, y=y, data=df1, palette=pal, bw=sigma, width_viol=0.6, ax=ax1,
                 orient=ort, move=0.2, pointplot=True, alpha=0.65)
    pt.RainCloud(x=x1, y=y, data=df1_50, palette=pal, bw=sigma, width_viol=0.6,
                 ax=ax1_50, orient=ort, move=0.2, pointplot=True, alpha=0.65)
    pt.RainCloud(x=x2, y=y, data=df2, palette=pal, bw=sigma, width_viol=0.6, ax=ax2,
                 orient=ort, move=0.2, pointplot=True, alpha=0.65)
    pt.RainCloud(x=x2, y=y, data=df2_50, palette=pal, bw=sigma, width_viol=0.6,
                 ax=ax2_50, orient=ort, move=0.2, pointplot=True, alpha=0.65)
    pt.RainCloud(x=x3, y=y, data=df3, palette=pal, bw=sigma, width_viol=0.6, ax=ax3,
                 orient=ort, move=0.2, pointplot=True, alpha=0.65)
    pt.RainCloud(x=x3, y=y, data=df3_50, palette=pal, bw=sigma, width_viol=0.6,
                 ax=ax3_50, orient=ort, move=0.2, pointplot=True, alpha=0.65)
    ax1.set_ylabel("Mean accuracy", fontsize=16)
    ax1.set_xlabel("Number of dendritic segments", fontsize=16)
    ax1_50.set_ylabel("Mean accuracy", fontsize=16)
    ax1_50.set_xlabel("Number of dendritic segments", fontsize=16)

    ax2.set(ylabel="")
    ax2.set_xlabel("Activation density", fontsize=16)

    ax2_50.set(ylabel="")
    ax2_50.set_xlabel("Activation density", fontsize=16)

    ax3.set(ylabel="")
    ax3.set_xlabel("FF Weight density", fontsize=16)
    ax3_50.set(ylabel="")
    ax3_50.set_xlabel("FF Weight density", fontsize=16)

    # Add 10 tasks and 50 tasks labels on the left
    plt.figtext(-0.02, 0.7, "10 TASKS", fontsize=16)
    plt.figtext(-0.02, 0.28, "50 TASKS", fontsize=16)

    fig.suptitle(
        """Impact of different hyperparameters on \n 10-tasks and 50-tasks
        permuted MNIST performance""",
        fontsize=16,
    )
    if savefigs:
        plt.savefig(
            f"{figs_dir}/hyperparameter_search_panel.png", bbox_inches="tight"
        )


def performance_across_tasks():
    """
    Similar representation as previous function (hyperparameter_search_panel)
    but in this case, it represents the performance along the number of tasks
    and we plot all the hyperparameters on the same figure for each figures.
    """

    df_path1 = f"{experiment_folder}segment_search_all.csv"
    df1 = pd.read_csv(df_path1)

    df_path2 = f"{experiment_folder}kw_sparsity_search_all.csv"
    df2 = pd.read_csv(df_path2)

    df_path3 = f"{experiment_folder}w_sparsity_search_all.csv"
    df3 = pd.read_csv(df_path3)

    df_path1_50 = f"{experiment_folder}segment_search_50_all.csv"
    df1_50 = pd.read_csv(df_path1_50)

    df_path2_50 = f"{experiment_folder}kw_sparsity_search_50_all.csv"
    df2_50 = pd.read_csv(df_path2_50)

    df_path3_50 = f"{experiment_folder}w_sparsity_search_50_all.csv"
    df3_50 = pd.read_csv(df_path3_50)

    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax1_50 = fig.add_subplot(gs[1, 0])
    ax2_50 = fig.add_subplot(gs[1, 1])
    ax3_50 = fig.add_subplot(gs[1, 2])

    x1 = "Iteration"
    hue1 = "Num segments"
    hue2 = "Activation sparsity"
    hue3 = "FF weight sparsity"
    y = "Accuracy"
    ort = "v"
    pal = sns.color_palette(n_colors=10)
    sigma = 0.2
    fig.suptitle(
        """Performance along number of tasks with different
        hyperpameter conditions""",
        fontsize=16,
    )

    pt.RainCloud(x=x1, y=y, hue=hue1, data=df1, palette=pal, bw=sigma, width_viol=0.6,
                 ax=ax1, orient=ort, move=0.2, pointplot=True, alpha=0.65)

    l, h = ax1.get_legend_handles_labels()
    ax1.legend(handles=l[0:10], labels=h[0:10], fontsize="8")

    pt.RainCloud(x=x1, y=y, hue=hue2, data=df2, palette=pal, bw=sigma, width_viol=0.6,
                 ax=ax2, orient=ort, move=0.2, pointplot=True, alpha=0.65)

    l, h = ax2.get_legend_handles_labels()
    ax2.legend(handles=l[0:9], labels=h[0:9], fontsize="8")

    pt.RainCloud(x=x1, y=y, hue=hue3, data=df3, palette=pal, bw=sigma, width_viol=0.6,
                 ax=ax3, orient=ort, move=0.2, pointplot=True, alpha=0.65)
    l, h = ax3.get_legend_handles_labels()
    ax3.legend(handles=l[0:8], labels=h[0:8], fontsize="8")

    pt.RainCloud(x=x1, y=y, hue=hue1, data=df1_50, palette=pal, bw=sigma,
                 width_viol=0.6, ax=ax1_50, orient=ort, move=0.2, pointplot=True,
                 alpha=0.65)
    l, h = ax1_50.get_legend_handles_labels()
    labels = h[0:9]
    ax1_50.legend(
        handles=l[0:9],
        labels=labels,
        fontsize="8",
    )

    pt.RainCloud(x=x1, y=y, hue=hue2, data=df2_50, palette=pal, bw=sigma,
                 width_viol=0.6, ax=ax2_50, orient=ort, move=0.2, pointplot=True,
                 alpha=0.65)
    l, h = ax2_50.get_legend_handles_labels()
    ax2_50.legend(handles=l[0:8], labels=h[0:8], fontsize="8")

    pt.RainCloud(x=x1, y=y, hue=hue3, data=df3_50, palette=pal, bw=sigma,
                 width_viol=0.6, ax=ax3_50, orient=ort, move=0.2, pointplot=True,
                 alpha=0.65)
    l, h = ax3_50.get_legend_handles_labels()
    ax3_50.legend(handles=l[0:8], labels=h[0:8], fontsize="8")

    ax1.set_xlabel("")
    ax1.set_ylabel("Mean Accuracy")
    ax1.set_title("Number of segments")
    ax1_50.set_xlabel("Tasks", fontsize=16)
    ax1_50.set_ylabel("Mean Accuracy")
    ax1_50.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1_50.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_title("Activation density")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2_50.set_xlabel("Tasks", fontsize=16)
    ax2_50.set_ylabel("")
    ax2_50.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2_50.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_title("FF weight density")
    ax3_50.set_xlabel("Tasks", fontsize=16)
    ax3_50.set_ylabel("")
    ax3_50.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax3_50.xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.figtext(-0.01, 0.7, "  10 TASKS", fontsize=14)
    plt.figtext(-0.01, 0.28, "  50 TASKS", fontsize=14)

    if savefigs:
        plt.savefig(f"{figs_dir}/hyperparameter_search_panel_along_tasks.png")


if __name__ == "__main__":

    savefigs = True
    figs_dir = "figs/"
    if savefigs:
        if not os.path.isdir(f"{figs_dir}"):
            os.makedirs(f"{figs_dir}")

    experiment_folder = "data_hyperparameter_search/"

    hyperparameter_search_panel()
    performance_across_tasks()
