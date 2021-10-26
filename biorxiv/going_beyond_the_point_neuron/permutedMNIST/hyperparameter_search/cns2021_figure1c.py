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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1)


def cns_figure_1c():
    """
    CNS 2021 abstract figure 1C.
    """
    data_folder = "cns2021_figure1c_data/"
    savefigs = True

    df_path1 = f"{data_folder}nb_segment_search2.csv"
    df1 = pd.read_csv(df_path1)

    df_path1bis = f"{data_folder}nb_segment_search3.csv"
    df1bis = pd.read_csv(df_path1bis)

    df_path2 = f"{data_folder}kw_sparsity_search.csv"
    df2 = pd.read_csv(df_path2)

    relevant_columns = ["Activation sparsity", "FF weight sparsity", "Num segments",
                        "Accuracy"]

    df1 = df1[relevant_columns]
    df1bis = df1bis[relevant_columns]
    df2 = df2[relevant_columns]

    df1 = pd.concat([df1, df1bis])

    gs = gridspec.GridSpec(1, 2)
    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    x1 = "Num segments"
    x2 = "Activation sparsity"

    y = "Accuracy"
    ort = "v"
    pal = "Set2"
    sigma = 0.2
    fig.suptitle(
        "Impact of the number of dendritic segments or the\n \
                 activation sparsity on 10-tasks permuted MNIST performance",
        fontsize=16,
    )

    pt.RainCloud(x=x1, y=y, data=df1, palette=pal, bw=sigma, width_viol=0.6, ax=ax1,
                 orient=ort, move=0.2, pointplot=True, alpha=0.65)
    pt.RainCloud(x=x2, y=y, data=df2, palette=pal, bw=sigma, width_viol=0.6, ax=ax2,
                 orient=ort, move=0.2, pointplot=True, alpha=0.65)
    ax1.set_ylim([0.9, 0.96])
    ax1.set_ylabel("Mean accuracy", fontsize=16)
    ax1.set_xlabel("Number of dendritic segments", fontsize=16)
    ax1.set_xticklabels(["2", "3", "5", "7", "10", "14", "20"], fontsize=14)
    ax1.set_yticklabels(
        ["0.90", "0.91", "0.92", "0.93", "0.94", "0.95", "0.96"], fontsize=14
    )
    ax2.set_ylim([0.60, 1])
    ax2.set_ylabel("")
    ax2.set_xticklabels(
        ["0.99", "0.95", "0.9", "0.8", "0.6", "0.4", "0.2", "0.1"], fontsize=14
    )
    ax2.set_yticklabels(["0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90",
                         "0.95", "1.0"], fontsize=14)
    ax2.set_xlabel("Activation sparsity", fontsize=16)

    if savefigs:
        plt.savefig("cns2021_figure1c.png", bbox_inches="tight", dpi=1200)


if __name__ == "__main__":
    cns_figure_1c()
