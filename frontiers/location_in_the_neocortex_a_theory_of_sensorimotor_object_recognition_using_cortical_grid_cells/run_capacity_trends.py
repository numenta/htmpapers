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

import argparse

import capacity_simulation


def experiment(bumpType, numModules=10, locationModuleWidth=10,
               numFeatures=100, thresholds=-1):
  return {
    "locationModuleWidth": locationModuleWidth,
    "cellCoordinateOffsets": (0.001, 0.999),
    "bumpType": bumpType,
    "initialIncrement": 128,
    "minAccuracy": 0.9,
    "capacityResolution": 1,
    "capacityPercentageResolution": -1.0,
    "featuresPerObject": 10,
    "objectWidth": 4,
    "numFeatures": numFeatures,
    "featureDistribution": "AllFeaturesEqual_Replacement",
    "useTrace": False,
    "noiseFactor": 0,
    "moduleNoiseFactor": 0,
    "numModules": numModules,
    "thresholds": thresholds,
    "seed1": -1,
    "seed2": -1,
    "anchoringMethod": "corners",
  }


def getExperiments(bumpType):
  return (
    [experiment(bumpType, numModules=numModules, thresholds=thresholds)
     for numModules in [5, 10, 15, 20, 25, 30, 35, 40]
     for thresholds in [-1, 0]]

    # Only test numModules=1 with a 100% threshold.
    # With the usual 80% threshold, it rounds up to 100% and is equivalent.
    # The results can be confusing, because with some parameters a single module
    # with 100% threshold has higher capacity than 5 modules with a 80%
    # threshold.
    +
    [experiment(bumpType, numModules=1, thresholds=0)]

    +
    [experiment(bumpType, locationModuleWidth=locationModuleWidth)
     for locationModuleWidth in range(2, 21)]

    +
    [experiment(bumpType, numFeatures=numUniqueFeatures)
     for numUniqueFeatures in [5, 10, 20, 50, 75, 100, 150, 200, 250, 300, 350,
                               400]]
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--resultName", type=str, required=True)
  parser.add_argument("--bumpType", type=str, required=True)
  parser.add_argument("--repeat", type=int, default=1)
  args = parser.parse_args()

  capacity_simulation.runExperiments(getExperiments(args.bumpType) * args.repeat,
                                     args.resultName,
                                     appendResults=True)
