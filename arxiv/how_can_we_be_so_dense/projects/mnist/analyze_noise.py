# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import logging
import os
from collections import OrderedDict, defaultdict
from os.path import basename

import numpy as np
from torchvision import datasets, transforms

from pytorch.image_transforms import RandomNoise

logging.basicConfig(level=logging.ERROR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from pytorch.mnist_sparse_experiment import MNISTSparseExperiment

NOISE_VALUES = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35",
                "0.4", "0.45", "0.5"]



def plotNoiseCurve(suite, values, results, plotPath, format):
  fig, ax = plt.subplots()
  fig.suptitle("Accuracy vs noise")
  ax.set_xlabel("Noise")
  ax.set_ylabel("Accuracy (percent)")
  for exp in results:
    values = suite.get_value(exp, 0, values, "last")
    df = pd.DataFrame.from_dict(values, orient='index')
    ax.plot(df["testerror"], **format[exp])

  # ax.xaxis.set_ticks(np.arange(0.0, 0.5 + 0.1, 0.1))
  plt.legend()
  plt.grid(axis='y')
  plt.savefig(plotPath)
  plt.close()



def plotDropoutByTotalCorrect(results, plotPath, format):
  fig, ax = plt.subplots()
  fig.suptitle("Dropout by Total Correct")
  ax.set_xlabel("Dropout")
  ax.set_ylabel("Total Correct")
  for exp in results:
    data = OrderedDict(sorted(results[exp].items(), key=lambda x: x[0]))
    ax.plot(data.keys(), data.values(), **format[exp])

  xticks = data.keys()
  ax.xaxis.set_ticks(np.arange(min(xticks), max(xticks) + 0.1, 0.1))
  plt.legend()
  plt.savefig(plotPath)
  plt.close()



def filterResults(results, filter):
  """
  Filter results containing the given condition
  :param results: list of experiments returned by `suite.get_exps`
  :param filter: list of conditions on the experiment parameters. For example:
                 ["dropout0.0", "dropout0.50"]
  :return: filtered results
  """
  return [exp for exp in results if any(map(lambda v: v in exp, filter))]



def configureNoisePlot(suite, experiments, labels, linestyles,
                       marker, filter=None):
  """
  Load experiment results anc configure the "Noise curve" Plot
  :param suite: The configured experiment suite. Must call `parse_opt` and `
                parse_cfg` before calling this functions
  :param experiments: list containing the experiments to load
  :param experiments: list containing the experiments to load
  :param labels: list containing the plot labels for each experiment
  :param linestyles: list containing the plot linestyle for each experiment
  :param filter: list containing the specific parameters to filter
  :return: tuple containing the experiment results and plot formats to be passed
           to `plotNoiseCurve` function
  """
  formats = dict()
  results = []
  for i in xrange(len(experiments)):
    path = suite.get_exp(experiments[i])[0]
    data = suite.get_exps(path=path)
    if filter is not None:
      data = filterResults(data, filter)

    # Format Linear Noise curve
    if len(data) > 1:
      format = {exp: {
        "label": "{},{}".format(labels[i], basename(exp)),
        "linestyle": "{}".format(linestyles[i]),
        "marker": "{}".format(marker[i])
      } for exp in data}
    else:
      format = {data[0]: {
        "label": "{}".format(labels[i]),
        "linestyle": "{}".format(linestyles[i]),
        "marker": "{}".format(marker[i])
      }}

    formats.update(format)
    results.extend(data)

  return (results, formats)



def configureDropoutByTotalCorrectPlot(suite, experiments, labels, linestyles):
  """
  Load experiment results anc configure the "Dropout By Total Correct" Plot
  :param suite: The configured experiment suite. Must call `parse_opt` and `
                parse_cfg` before calling this functions
  :param experiments: list containing the experiments to load
  :param labels: list containing the plot labels for each experiment
  :param linestyles: list containing the plot linestyle for each experiment
  :return: tuple containing the experiment results and plot formats to be passed
           to `plotDropoutByTotalCorrect` function
  """
  results = defaultdict(dict)
  formats = dict()
  for i in xrange(len(experiments)):
    experiment = experiments[i]
    formats[experiment] = {"label": labels[i], "linestyle": linestyles[i]}

    path = suite.get_exp(experiment)[0]
    data = suite.get_exps(path=path)
    for exp in data:
      dropout = suite.get_params(exp)["dropout"]
      totalCorrect = suite.get_value(exp, 0, "totalCorrect", "last")
      results[experiment][dropout] = totalCorrect

  return (results, formats)



def plotImagesWithNoise(suite, noise_values, plotPath):
  """
  Plot Sample MNIST images with noise
  :param suite: The configured experiment suite. Must call `parse_opt` and `
                parse_cfg` before calling this functions
  :param noise: list of noise values to plot
  """
  datadir = suite.cfgparser.defaults()["datadir"]

  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
  dataset = datasets.MNIST(datadir, train=False, download=True,
                           transform=transform)

  num_noise = len(noise_values)
  fig = plt.figure(figsize=(num_noise, 4))
  for y in range(4):
    for x in range(num_noise):
      transform.transforms.append(
        RandomNoise(noise_values[x], whiteValue=0.1307 + 2 * 0.3081))
      img, _ = dataset[y]
      transform.transforms.pop()

      ax = fig.add_subplot(4, num_noise, y * num_noise + x + 1)
      ax.set_axis_off()
      ax.imshow(img.numpy().reshape((28, 28)), cmap='gray')
      if y == 0:
        ax.set_title("{0}%".format(noise_values[x] * 100))

  plt.tight_layout()
  plt.savefig(plotPath)
  plt.close()



if __name__ == '__main__':
  # Initialize experiment options and parameters
  suite = MNISTSparseExperiment()
  suite.parse_opt()
  suite.parse_cfg()
  path = suite.cfgparser.defaults()['path']

  # Plot Noise Curve (MNIST)
  results, format = configureNoisePlot(suite,
                                       experiments=[
                                         "denseCNN1",
                                         "denseCNN2",
                                         "sparseCNN1",
                                         "sparseCNN2",
                                       ],
                                       labels=["dense-CNN1", "dense-CNN2",
                                               "sparse-CNN1", "sparse-CNN2"],
                                       linestyles=["--", "--", "-", "-"],
                                       marker=["o", "x", "*", "x"],
                                       )

  plotPath = os.path.join(path, "accuracy_vs_noise.pdf")
  plotNoiseCurve(suite=suite, values=NOISE_VALUES, results=results, format=format,
                 plotPath=plotPath)

  plotPath = os.path.join(path, "mnist_images_with_noise.pdf")
  plotImagesWithNoise(suite=suite, noise_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                      plotPath=plotPath)
