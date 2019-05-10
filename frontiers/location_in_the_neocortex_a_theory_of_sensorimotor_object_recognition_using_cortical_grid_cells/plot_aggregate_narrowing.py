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

"""Plot location module representations during narrowing."""

import argparse
from collections import defaultdict
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def aggregateChart(inFilename, outFilename, objectCounts, ylim):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  markers = ["D", "v", "o", "^"]
  markersizes = [3, 4, 4, 4]

  plt.figure(figsize=(3.25, 2.5), tight_layout = {"pad": 0})

  cellsPerModule = 100
  numSteps = 9

  resultsByNumObjects = defaultdict(list)
  with open(inFilename, "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    numObjects = exp[0]["numObjects"]
    timestepsByObject = exp[1]["locationLayerTimelineByObject"].values()

    meanDensityByTimestep = [
      np.mean([len(module["activeCells"])
               for modules in timestepByObject
               for module in modules]) / float(cellsPerModule)
      for timestepByObject in zip(*timestepsByObject)
    ]

    resultsByNumObjects[numObjects].append(meanDensityByTimestep)

  percentiles = [5, 50, 95]

  for numObjects, marker, markersize in zip(objectCounts, markers, markersizes):
    trials = resultsByNumObjects[numObjects]

    x = []
    y = []
    errBelow = []
    errAbove = []

    for step, densityByTrial in zip(xrange(numSteps), zip(*trials)):
      x.append(step + 1)

      p1, p2, p3 = np.percentile(densityByTrial, percentiles)
      y.append(p2)
      errBelow.append(p2 - p1)
      errAbove.append(p3 - p2)

    plt.errorbar(x, y, yerr=[errBelow, errAbove], fmt="{}-".format(marker),
                 label="{} learned objects".format(numObjects), capsize=2,
                 markersize=markersize)

  plt.xlabel("Number of Sensations")
  plt.ylabel("Mean Cell Activation Density")

  plt.ylim(ylim)

  # Remove the errorbars from the legend.
  handles, labels = plt.gca().get_legend_handles_labels()
  handles = [h[0] for h in handles]

  # If there's any opacity, when we export a copy of this from Illustrator, it
  # creates a PDF that isn't compatible with Word.
  framealpha = 1.0

  plt.legend(handles, labels, framealpha=framealpha)

  filename = os.path.join(CHART_DIR, outFilename)
  print "Saving", filename
  plt.savefig(filename)


if __name__ == "__main__":
  plt.rc("font",**{"family": "sans-serif",
                   "sans-serif": ["Arial"],
                   "size": 8})

  parser = argparse.ArgumentParser()
  parser.add_argument("--inFile", type=str, required=True)
  parser.add_argument("--outFile", type=str, required=True)
  parser.add_argument("--objectCounts", type=int, nargs="+", default=[50, 75, 100, 125])
  parser.add_argument("--ylim", type=float, nargs=2, default=(-0.05, 1.05))
  args = parser.parse_args()

  aggregateChart(args.inFile, args.outFile, args.objectCounts, args.ylim)
