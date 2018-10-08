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

"""Plot capacity charts."""

import argparse
from collections import defaultdict
import json
import math
import os
import itertools

import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np

from htmresearch.frameworks.location import ambiguity_index

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def createCharts(inFilename, outFilename, squeezeLegend,
                 moduleWidths=(6, 10, 14),
                 moduleCounts=(5, 10, 15)):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  recognitionTimeResults = defaultdict(lambda: defaultdict(list))
  capacityResults = defaultdict(lambda: defaultdict(list))

  with open(inFilename, "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    moduleWidth = exp[0]["locationModuleWidth"]
    numObjects = exp[0]["numObjects"]

    numModules = exp[0]["numModules"]

    recognitionTimes = []
    for numSensationsStr, numOccurrences in exp[1]["convergence"].items():
      if numSensationsStr == "null":
        recognitionTimes += [np.inf] * numOccurrences
      else:
        recognitionTimes += [int(numSensationsStr)] * numOccurrences
    recognitionTimeResults[(moduleWidth, numModules)][numObjects] += recognitionTimes

    failed = exp[1]["convergence"].get("null", 0)
    accuracy = 1.0 - (float(failed) / float(numObjects))
    capacityResults[(moduleWidth, numModules)][numObjects].append(accuracy)

  fig, (axRecognitionTime, axCapacity) = plt.subplots(figsize=(3.25, 4.5),
                                                      nrows=2, sharex=True,
                                                      tight_layout = {"pad": 0})

  #
  # CAPACITY
  #

  colors = ("C0", "C1", "C2")
  markers = ("o", "o", "o")
  markerSizes = (1.5, 3, 4.5)
  for moduleWidth, color in reversed(zip(moduleWidths, colors)):
    for numModules, marker, markerSize in zip(moduleCounts, markers, markerSizes):
      resultsByNumObjects = capacityResults[(moduleWidth, numModules)]
      expResults = [(numObjects, sum(results) / len(results))
                     for numObjects, results in resultsByNumObjects.iteritems()]

      x = []
      y = []
      for i, j in sorted(expResults):
        x.append(i)
        y.append(j)

      axCapacity.plot(
        x, y, "{}-".format(marker), color=color, linewidth=1, markersize=markerSize
      )

  axCapacity.set_xlabel("Number of Learned Objects")
  axCapacity.set_ylabel("Recognition Accuracy After\nMany Sensations")

  #
  # RECOGNITION TIME
  #

  for moduleWidth, color in reversed(zip(moduleWidths, colors)):
    for numModules, marker, markerSize in zip(moduleCounts, markers, markerSizes):
      resultsByNumObjects = recognitionTimeResults[(moduleWidth, numModules)]

      expResults = sorted((numObjects, np.median(results))
                          for numObjects, results in resultsByNumObjects.iteritems())

      # Results up to the final non-infinite median.
      lineResults = [(numObjects, median)
                     for numObjects, median in expResults
                     if median != np.inf]

      # Results excluding the final non-infinite median.
      numCircleMarkers = len(lineResults)
      if len(lineResults) < len(expResults):
        numCircleMarkers -= 1

      axRecognitionTime.plot([numObjects for numObjects, median in lineResults],
               [median for numObjects, median in lineResults],
               "{}-".format(marker), markevery=xrange(numCircleMarkers),
               color=color, linewidth=1, markersize=markerSize)

      if (len(lineResults) < len(expResults) and len(lineResults) > 0):
        endNumObjects, endMedian = lineResults[-1]
        axRecognitionTime.plot([endNumObjects], [endMedian], "x", color=color,
                 markeredgewidth=markerSize/2, markersize=markerSize*1.5)

  axRecognitionTime.set_ylim(0, axRecognitionTime.get_ylim()[1])
  # axRecognitionTime.set_xlabel("# learned objects")
  axRecognitionTime.set_ylabel("Median Number of Sensations\nTo Recognition")

  if squeezeLegend:
    # If there's any opacity, when we export a copy of this from Illustrator, it
    # creates a PDF that isn't compatible with Word.
    framealpha = 1.0
    leg = axRecognitionTime.legend(loc="upper right", title="Cells per\n module",
                     # bbox_to_anchor=(1.035, 1.0),
                     frameon=True,
                     framealpha=framealpha,
                     handles=[matplotlib.lines.Line2D([], [], color=color)
                              for color in colors],
                     labels=["{}x{}".format(moduleWidth, moduleWidth)
                             for moduleWidth in moduleWidths])
    axRecognitionTime.add_artist(leg)


    leg = axRecognitionTime.legend(loc="lower right", title=" Number of \n   modules",
                     # bbox_to_anchor=(1.0, 0.5),
                     frameon=True,
                     borderpad=0.55,
                     framealpha=framealpha,
                     handles=[matplotlib.lines.Line2D([], [],
                                                      marker=marker,
                                                      markersize=markerSize,
                                                      color="black")
                              for marker, markerSize in zip(markers, markerSizes)],
                     labels=moduleCounts)
  else:
    leg = axRecognitionTime.legend(loc="upper right", title="Cells Per Module:       ",
                     bbox_to_anchor=(1.035, 1.0),
                     frameon=False,
                     handles=[matplotlib.lines.Line2D([], [], color=color)
                              for color in colors],
                     labels=["{}x{}".format(moduleWidth, moduleWidth)
                             for moduleWidth in moduleWidths])
    axRecognitionTime.add_artist(leg)


    leg = axRecognitionTime.legend(loc="center right", title="Number of Modules:",
                     bbox_to_anchor=(1.0, 0.6),
                     frameon=False,
                     handles=[matplotlib.lines.Line2D([], [],
                                                      marker=marker,
                                                      markersize=markerSize,
                                                      color="black")
                              for marker, markerSize in zip(markers, markerSizes)],
                     labels=moduleCounts)

  locs, labels = ambiguity_index.getTotalExpectedOccurrencesTicks_2_5(
      ambiguity_index.numOtherOccurrencesOfMostUniqueFeature_lowerBound50_100features_10locationsPerObject)
  locs = [loc for loc in locs
          if loc < axRecognitionTime.get_xlim()[1] - 10]

  ax2RecognitionTime = axRecognitionTime.twiny()
  ax2RecognitionTime.set_xlabel("Median Number of Locations Recalled by\nan Object's Rarest Feature", labelpad=8)
  ax2RecognitionTime.set_xticks(locs)
  ax2RecognitionTime.set_xticklabels(labels)
  ax2RecognitionTime.set_xlim(axRecognitionTime.get_xlim())

  ax2Capacity = axCapacity.twiny()
  ax2Capacity.set_xticks(locs)
  ax2Capacity.set_xticklabels(labels)
  ax2Capacity.set_xlim(axCapacity.get_xlim())
  ax2Capacity.tick_params(axis="x", which="major", pad=0)

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
  parser.add_argument("--squeezeLegend", action="store_true")
  args = parser.parse_args()

  createCharts(args.inFile, args.outFile, args.squeezeLegend)
