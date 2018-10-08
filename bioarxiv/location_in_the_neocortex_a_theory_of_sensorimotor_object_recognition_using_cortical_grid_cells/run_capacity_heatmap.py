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


def experiment(bumpType, locationModuleWidth=10, numFeatures=100):
  return {
    "locationModuleWidth": locationModuleWidth,
    "bumpType": bumpType,
    "cellCoordinateOffsets": (0.001, 0.999),
    "initialIncrement": 128,
    "minAccuracy": 0.9,
    "capacityResolution": 1,
    "capacityPercentageResolution": 0.02,
    "featuresPerObject": 10,
    "objectWidth": 4,
    "numFeatures": numFeatures,
    "featureDistribution": "AllFeaturesEqual_Replacement",
    "useTrace": False,
    "noiseFactor": 0,
    "moduleNoiseFactor": 0,
    "numModules": 10,
    "thresholds": -1,
    "seed1": -1,
    "seed2": -1,
    "anchoringMethod": "corners",
  }


def getExperiments(bumpType):
  locationModuleWidths = [6, 8, 10, 14, 17, 20]
  featureCounts = [w**2 for w in locationModuleWidths]

  return (
    [experiment(bumpType, locationModuleWidth=locationModuleWidth,
                numFeatures=numFeatures)
     for locationModuleWidth in locationModuleWidths
     for numFeatures in featureCounts]
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--resultName", type=str, required=True)
  parser.add_argument("--bumpType", type=str, required=True)
  parser.add_argument("--repeat", type=int, default=1)
  args = parser.parse_args()

  capacity_simulation.runExperiments(getExperiments(args.bumpType) * args.repeat,
                                     args.resultName, appendResults=True)
