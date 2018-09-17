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

"""Plot capacity trend charts."""

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



def createChart(inFilename, outFilename, modulesYmax, label1Position,
                label2Position, label3Position):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)


  capacitiesByParams = defaultdict(list)
  codesAreUnique = defaultdict(lambda: True)
  moduleCounts = set()
  allCellCounts = set()
  allFeatureCounts = set()
  with open(inFilename, "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    numModules = exp[0]["numModules"]
    thresholds = exp[0]["thresholds"]
    locationModuleWidth = exp[0]["locationModuleWidth"]
    numUniqueFeatures = exp[0]["numFeatures"]

    cellsPerModule = locationModuleWidth*locationModuleWidth

    moduleCounts.add(numModules)
    allCellCounts.add(cellsPerModule)
    allFeatureCounts.add(numUniqueFeatures)

    params = (numModules, cellsPerModule, thresholds, numUniqueFeatures)
    if "allLocationsAreUnique" in exp[1]:
      codesAreUnique[params] = codesAreUnique[params] and exp[1]["allLocationsAreUnique"]
    capacitiesByParams[params].append(exp[1]["numObjects"])

  moduleCounts = sorted(moduleCounts)
  allCellCounts = sorted(allCellCounts)
  allFeatureCounts = sorted(allFeatureCounts)

  meanCapacityByParams = {}
  for params, capacities in capacitiesByParams.iteritems():
    meanCapacityByParams[params] = sum(capacities) / float(len(capacities))


  fig, (ax1, ax2, ax3) = plt.subplots(figsize=(3.25, 1.35), ncols=3,
                                      sharey=True, tight_layout = {"pad": 0})

  #
  # NUMBER OF MODULES
  #
  cellsPerModule = 100
  numUniqueFeatures = 100
  markers = ["o", "*"]
  markerSizes = [2.0, 3.0]
  for thresholds, marker, markerSize in zip([-1, 0], markers, markerSizes):
    x = []
    y = []
    for numModules in moduleCounts:
      params = (numModules, cellsPerModule, thresholds, numUniqueFeatures)
      if params in meanCapacityByParams:
        x.append(numModules)
        y.append(meanCapacityByParams[params])

    ax1.plot(x, y, "{}-".format(marker), color="C0",
             markersize=markerSize,
             markevery=[i
                        for i, numModules in enumerate(x)
                        if codesAreUnique[(numModules,
                                           cellsPerModule,
                                           thresholds,
                                           numUniqueFeatures)]])

    ax1.plot(x, y, "x", markersize=4, markeredgewidth=2, color="red",
             markevery=[i
                        for i, numModules in enumerate(x)
                        if not codesAreUnique[(numModules,
                                               cellsPerModule,
                                               thresholds,
                                               numUniqueFeatures)]])

  ax1.text(label1Position[0], label1Position[1], "Threshold:")
  ax1.text(label2Position[0], label2Position[1], "$ n $")
  ax1.text(label3Position[0], label3Position[1], "$ n * 0.8 $")

  ax1.set_xlabel("Number of\nModules")
  ax1.set_ylabel("Capacity")
  ax1.set_ylim(0,
               (modulesYmax
                if modulesYmax is not None
                else ax1.get_ylim()[1]))
  xticks = range(0, max(moduleCounts) + 1, 5)
  ax1.set_xticks(xticks)
  ax1.set_xticklabels([(x if x % 20 == 0 else "")
                       for x in xticks])
  yticks = range(0, 501, 100)
  ax1.set_yticks(yticks)

  #
  # CELLS PER MODULE
  #
  numModules = 10
  thresholds = -1
  numUniqueFeatures = 100
  ax2.plot(allCellCounts, [meanCapacityByParams[(numModules,
                                                 cellsPerModule,
                                                 thresholds,
                                                 numUniqueFeatures)]
                           for cellsPerModule in allCellCounts],
           "o-", color="C0", markersize=2.0)

  ax2.set_xlabel("Cells Per\nModule")

  xticks = range(0, 401, 50)
  ax2.set_xticks(xticks)
  ax2.set_xticklabels([(x if x % 200 == 0 else "")
                       for x in xticks])


  #
  # NUMBER OF UNIQUE FEATURES
  #
  numModules = 10
  cellsPerModule = 100
  thresholds = -1
  ax3.plot(allFeatureCounts, [meanCapacityByParams[(numModules,
                                                    cellsPerModule,
                                                    thresholds,
                                                    numUniqueFeatures)]
                              for numUniqueFeatures in allFeatureCounts],
           "o-", color="C1", markersize=2.0)

  ax3.set_xlabel("Number of\nUnique Features")
  # ax3.set_xlim(0, ax3.get_xlim()[1])

  xticks = range(0, 401, 50)
  ax3.set_xticks(xticks)
  ax3.set_xticklabels([(x if x % 200 == 0 else "")
                       for x in xticks])

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
  parser.add_argument("--modulesYmax", type=float, default=None)
  parser.add_argument("--label1Position", type=float, nargs=2, default=(1, 685))
  parser.add_argument("--label2Position", type=float, nargs=2, default=(32, 590))
  parser.add_argument("--label3Position", type=float, nargs=2, default=(23, 200))
  args = parser.parse_args()

  createChart(args.inFile, args.outFile, args.modulesYmax, args.label1Position,
              args.label2Position, args.label3Position)
