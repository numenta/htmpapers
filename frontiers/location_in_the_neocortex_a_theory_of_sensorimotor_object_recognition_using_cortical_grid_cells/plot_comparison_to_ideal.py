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


def createChart(inFilename, outFilename, locationModuleWidths, legendPosition):

  numSteps = 12

  resultsByParams = defaultdict(lambda: defaultdict(list))

  with open(inFilename, "r") as f:
    experiments = json.load(f)

  for exp in experiments:
    locationModuleWidth = exp[0]["locationModuleWidth"]
    cum = 0
    for i in xrange(40):
      step = i + 1
      count = exp[1]["convergence"].get(str(step), 0)
      resultsByParams[locationModuleWidth][step].append(count)

    count = exp[1]["convergence"].get("null", 0)
    step = -1
    resultsByParams[locationModuleWidth][step].append(count)

  with open("results/ideal.json", "r") as f:
    idealResults = json.load(f)

  with open("results/bof.json", "r") as f:
    bofResults = json.load(f)

  plt.figure(figsize=(3.25, 2.5), tight_layout = {"pad": 0})

  for yData, label, fmt in [(resultsByParams[locationModuleWidth],
                             "{}x{} Cells Per Module".format(locationModuleWidth,
                                                             locationModuleWidth),
                             fmt)
                            for locationModuleWidth, fmt in zip(locationModuleWidths,
                                                                ["s-", "o-", "^-"])]:
    x = [i+1 for i in xrange(numSteps)]
    y = []
    tot = float(sum([sum(counts)
                     for counts in yData.values()]))
    cum = 0.0
    for step in x:
      if step in yData:
        counts = yData[step]
      else:
        print yData
        counts = yData[str(step)]
      cum += float(sum(counts))
      y.append(cum / tot)
    std = [np.std(yData[step]) for step in x]
    yBelow = [yi - stdi for yi, stdi in zip(y, std)]
    yAbove = [yi + stdi for yi, stdi in zip(y, std)]

    plt.plot(
        x, y, fmt, label=label,
    )
    #plt.fill_between(x, yBelow, yAbove, alpha=0.3)


  for results, label, fmt, markersize in [(idealResults, "Ideal Observer", "x--", 10),
                                          (bofResults, "Bag of Features", "d--", None)]:
    x = [i+1 for i in xrange(numSteps)]
    y = []
    std = [np.std(results.get(str(steps), [0])) for steps in x]
    tot = float(sum([sum(counts) for counts in results.values()]))
    cum = 0.0
    for steps in x:
      counts = results.get(str(steps), [])
      if len(counts) > 0:
        cum += float(sum(counts))
      y.append(cum / tot)
    yBelow = [yi - stdi for yi, stdi in zip(y, std)]
    yAbove = [yi + stdi for yi, stdi in zip(y, std)]
    plt.plot(
        x, y, fmt, label=label, markersize=markersize
    )
    #plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  # Formatting
  plt.xlabel("Number of Sensations")
  plt.ylabel("Cumulative Accuracy")

  plt.xticks([(i+1) for i in xrange(numSteps)])

  # If there's any opacity, when we export a copy of this from Illustrator, it
  # creates a PDF that isn't compatible with Word.
  framealpha = 1.0
  plt.legend(loc="center right", bbox_to_anchor=legendPosition,
             framealpha=framealpha)

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
  parser.add_argument("--locationModuleWidth", type=int, nargs=3,
                      default=[17, 20, 40])
  parser.add_argument("--legendPosition", type=float, nargs=2, default=None)
  args = parser.parse_args()

  createChart(args.inFile, args.outFile, args.locationModuleWidth, args.legendPosition)
