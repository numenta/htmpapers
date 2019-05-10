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

"""Plot convergence chart that compares different algorithms."""

import argparse
from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt
import numpy as np

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def getCumulativeAccuracy(convergenceFrequencies):
  tot = float(sum(convergenceFrequencies.values()))

  results = []

  cum = 0.0
  for step in xrange(1, 41):
    cum += convergenceFrequencies.get(str(step), 0)
    results.append(cum / tot)

  return results


def createChart(inFilename, outFilename, locationModuleWidths, legendPosition):

  numSteps = 12

  resultsByParams = defaultdict(list)

  with open(inFilename, "r") as f:
    experiments = json.load(f)

  for exp in experiments:
    locationModuleWidth = exp[0]["locationModuleWidth"]
    resultsByParams[locationModuleWidth].append(
      getCumulativeAccuracy(exp[1]["convergence"]))

  with open("results/ideal.json", "r") as f:
    idealResults = [getCumulativeAccuracy(trial)
                    for trial in json.load(f)]

  with open("results/bof.json", "r") as f:
    bofResults = [getCumulativeAccuracy(trial)
                  for trial in json.load(f)]

  plt.figure(figsize=(3.25, 2.5), tight_layout = {"pad": 0})

  
  data = (
    [(idealResults, "Ideal Observer", "x--", 10, 1)]

    +
    [(resultsByParams[locationModuleWidth],
      "{}x{} Cells Per Module".format(locationModuleWidth,
                                      locationModuleWidth),
      fmt,
      None,
      0)
     for locationModuleWidth, fmt in zip(locationModuleWidths,
                                         ["s-", "o-", "^-"])]

    +
    [(bofResults, "Bag of Features", "d--", None, -1)])

  percentiles = [5, 50, 95]

  for resultsByTrial, label, fmt, markersize, zorder in data:
    x = []
    y = []
    errBelow = []
    errAbove = []

    resultsByStep = zip(*resultsByTrial)

    for step, results in zip(xrange(numSteps), resultsByStep):
      x.append(step + 1)
      p1, p2, p3 = np.percentile(results, percentiles)
      y.append(p2)
      errBelow.append(p2 - p1)
      errAbove.append(p3 - p2)

    plt.errorbar(x, y, yerr=[errBelow, errAbove], fmt=fmt, label=label,
                 capsize=2, markersize=markersize, zorder=zorder)

  # Formatting
  plt.xlabel("Number of Sensations")
  plt.ylabel("Cumulative Accuracy")

  plt.xticks([(i+1) for i in xrange(numSteps)])

  # Remove the errorbars from the legend.
  handles, labels = plt.gca().get_legend_handles_labels()
  handles = [h[0] for h in handles]

  plt.legend(handles, labels, loc="center right", bbox_to_anchor=legendPosition)

  outFilePath = os.path.join(CHART_DIR, outFilename)
  print "Saving", outFilePath
  plt.savefig(outFilePath)

  plt.clf()


if __name__ == "__main__":
  plt.rc("font",**{"family": "sans-serif",
                   "sans-serif": ["Arial"],
                   "size": 8})

  parser = argparse.ArgumentParser()
  parser.add_argument("--inFile", type=str, required=True)
  parser.add_argument("--outFile", type=str, required=True)
  parser.add_argument("--locationModuleWidth", type=int, nargs='+',
                      default=[17, 20, 40])
  parser.add_argument("--legendPosition", type=float, nargs=2, default=None)
  args = parser.parse_args()

  createChart(args.inFile, args.outFile, args.locationModuleWidth, args.legendPosition)
