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


def experiment(bumpType, numFeatures=100, featuresPerObject=10,
               featureDistribution="AllFeaturesEqual_Replacement",
               numObjects=42):
  return {
    "locationModuleWidth": 10,
    "cellCoordinateOffsets": (0.001, 0.999),
    "bumpType": bumpType,
    "numObjects": numObjects,
    "featuresPerObject": featuresPerObject,
    "objectWidth": 4,
    "numFeatures": numFeatures,
    "featureDistribution": featureDistribution,
    "useTrace": False,
    "useRawTrace": False,
    "logCellActivity": False,
    "logNumFeatureOccurrences": True,
    "noiseFactor": 0,
    "moduleNoiseFactor": 0,
    "numModules": 10,
    "numSensations": -1,
    "thresholds": -1,
    "seed1": -1,
    "seed2": -1,
    "anchoringMethod": "corners",
  }

def getExperiments(bumpType):
  return (
    # Baseline
    [experiment(bumpType, numObjects=numObjects)
     for numObjects in range(50, (401 if bumpType == "square" else 301), 50)]

    # Baseline, 40 unique features
    +
    [experiment(bumpType, numObjects=numObjects, numFeatures=40)
     for numObjects in range(50, (251 if bumpType == "square" else 151), 50)]

    # Baseline, 5 features per object
    +
    [experiment(bumpType, numObjects=numObjects, featuresPerObject=5)
     for numObjects in range(50, (701 if bumpType == "square" else 501), 50)]

    # Baseline, no replacement
    +
    [experiment(bumpType, numObjects=numObjects,
                featureDistribution="AllFeaturesEqual_NoReplacement")
     for numObjects in ([50, 100, 150, 200, 225, 250, 275, 300, 350]
                        if bumpType == "square"
                        else [50, 75, 100, 125, 150, 175])]

    # Baseline, but with two pools
    +
    [experiment(bumpType, numObjects=numObjects,
                featureDistribution="TwoPools_Replacement")
     for numObjects in range(50, (551 if bumpType == "square" else 401), 50)]

    # Two pools, structured
    +
    [experiment(bumpType, numObjects=numObjects,
                featureDistribution="TwoPools_Structured")
     for numObjects in range(50, (1001 if bumpType == "square" else 701),
                             50)]
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
