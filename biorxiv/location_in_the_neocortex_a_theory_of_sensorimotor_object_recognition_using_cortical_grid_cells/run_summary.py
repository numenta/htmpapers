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

import convergence_simulation


def experiment(bumpType, numModules=10, locationModuleWidth=10, numObjects=42):
  return {
    "locationModuleWidth": locationModuleWidth,
    "cellCoordinateOffsets": (0.001, 0.999),
    "bumpType": bumpType,
    "numObjects": numObjects,
    "featuresPerObject": 10,
    "objectWidth": 4,
    "numFeatures": 100,
    "featureDistribution": "AllFeaturesEqual_Replacement",
    "useTrace": False,
    "useRawTrace": False,
    "logCellActivity": False,
    "logNumFeatureOccurrences": False,
    "noiseFactor": 0,
    "moduleNoiseFactor": 0,
    "numModules": numModules,
    "numSensations": -1,
    "thresholds": -1,
    "seed1": -1,
    "seed2": -1,
    "anchoringMethod": "corners",
  }


def getExperiments(bumpType):
  return (
    [experiment(bumpType, locationModuleWidth=6, numObjects=numObjects,
                numModules=numModules)
     for numObjects in [3] + range(20, (221 if bumpType == "square" else 161),
                                   20)
     for numModules in [5, 10, 15]]

    +
    [experiment(bumpType, locationModuleWidth=10, numObjects=numObjects,
                numModules=numModules)
     for numObjects in [3] + range(20, (381 if bumpType == "square" else 281),
                                   20)
     for numModules in [5, 10, 15]]

    +
    [experiment(bumpType, locationModuleWidth=14, numObjects=numObjects,
                numModules=numModules)
     for numObjects in [3] + range(20, (681 if bumpType == "square" else 421),
                                   20)
     for numModules in [5, 10, 15]]
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--resultName", type=str, required=True)
  parser.add_argument("--bumpType", type=str, required=True)
  parser.add_argument("--repeat", type=int, default=1)
  args = parser.parse_args()

  convergence_simulation.runExperiments(getExperiments(args.bumpType) * args.repeat,
                                        args.resultName,
                                        appendResults=True)
