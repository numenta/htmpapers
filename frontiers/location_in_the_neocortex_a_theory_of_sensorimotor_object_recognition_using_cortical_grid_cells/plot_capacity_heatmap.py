# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

"""Plot comparison chart."""

import argparse
from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy.optimize

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def chart2(inFilename, outFilename, cellCounts, featureCounts):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  capacitiesByParams = defaultdict(list)
  with open(inFilename, "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    locationModuleWidth = exp[0]["locationModuleWidth"]
    numUniqueFeatures = exp[0]["numFeatures"]

    cellsPerModule = locationModuleWidth*locationModuleWidth

    capacitiesByParams[(cellsPerModule, numUniqueFeatures)].append(exp[1]["numObjects"])

  meanCapacityByParams = {}
  for params, capacities in capacitiesByParams.iteritems():
    meanCapacityByParams[params] = sum(capacities) / float(len(capacities))

  xlabels = [str(v) for v in featureCounts]
  ylabels = [str(v) for v in cellCounts]

  plotData = np.empty((len(cellCounts), len(featureCounts)), dtype="float")
  for i, cellsPerModule in enumerate(cellCounts):
    for j, numUniqueFeatures in enumerate(featureCounts):
      plotData[i, j] = meanCapacityByParams[(cellsPerModule, numUniqueFeatures)]

  fig, ax = plt.subplots(figsize=(3.25, 3.25), tight_layout = {"pad": 0})

  # Customize vmax so that the colors stay suffiently dark so that the white
  # text is readable.
  plt.imshow(plotData,
             norm=colors.LogNorm(vmin=plotData.min(), vmax=plotData.max()*3.0))

  ax.xaxis.set_label_position('top')
  ax.xaxis.tick_top()
  ax.set_xticks(np.arange(len(xlabels)))
  ax.set_yticks(np.arange(len(ylabels)))
  ax.set_xticklabels(xlabels)
  ax.set_yticklabels(ylabels)
  plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

  for i in xrange(len(ylabels)):
    for j in xrange(len(xlabels)):
      text = ax.text(j, i, str(int(plotData[i, j])), ha="center", va="center",
                     color="w")

  plt.xlabel("Number of Unique Features")
  plt.ylabel("Cells Per Module")

  filePath = os.path.join(CHART_DIR, outFilename)
  print "Saving", filePath
  plt.savefig(filePath)


if __name__ == "__main__":
  plt.rc("font",**{"family": "sans-serif",
                   "sans-serif": ["Arial"],
                   "size": 8})

  parser = argparse.ArgumentParser()
  parser.add_argument("--inFile", type=str, required=True)
  parser.add_argument("--outFile", type=str, required=True)
  args = parser.parse_args()

  counts = [w**2 for w in [6, 8, 10, 14, 17, 20]]

  chart2(args.inFile, args.outFile,
         cellCounts=counts,
         featureCounts=counts)
