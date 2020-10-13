#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

''' 
This is a basic script to perform object recognition with SDRs 
generated from a CNN
'''


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

"""
Measure how many objects the model can learn and recognize with sufficient
accuracy.
"""

import argparse
import io
import math
import os
import random
from multiprocessing import cpu_count, Pool
from copy import copy
import time
import json


import numpy as np

from htmresearch.frameworks.location.object_generation import generateObjects
from custom_object_creation import (
  PIUNCorticalColumn, PIUNExperiment, PIUNExperiment_original, PIUNExperimentMonitor)
from htmresearch.frameworks.location.two_layer_tracing import (
  PIUNVisualizer as trace)

from pre_process_SDRs import generate_image_objects

# SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# class PIUNCellActivityTracer(PIUNExperimentMonitor):
#   def __init__(self, exp):
#     self.exp = exp
#     self.locationLayerTimelineByObject = {}
#     self.inferredStepByObject = {}
#     self.currentObjectName = None

#   def afterLocationAnchor(self, **kwargs):
#     self.locationLayerTimelineByObject[self.currentObjectName].append(
#       [module.sensoryAssociatedCells.tolist()
#        for module in self.exp.column.L6aModules])

#   def beforeInferObject(self, obj):
#     self.currentObjectName = obj["name"]
#     self.locationLayerTimelineByObject[obj["name"]] = []

#   def afterInferObject(self, obj, inferredStep):
#     self.inferredStepByObject[obj["name"]] = inferredStep


numOffsets = 2
cellCoordinateOffsets = tuple([i * (0.998 / (numOffsets-1)) + 0.001
                               for i in xrange(numOffsets)])


def basic_object_learning(train_only_bool,
                 locationModuleWidth,
                 bumpType,
                 cellCoordinateOffsets,
                 initialIncrement,
                 minAccuracy,
                 capacityResolution,
                 capacityPercentageResolution,
                 featuresPerObject,
                 objectWidth,
                 numObjects,
                 numFeatures,
                 featureDistribution,
                 noiseFactor,
                 moduleNoiseFactor,
                 numModules,
                 thresholds,
                 seed1,
                 seed2,
                 anchoringMethod):

  if seed1 != -1:
    np.random.seed(seed1)

  if seed2 != -1:
    random.seed(seed2)


  locationConfigs = []
  scale = 40.0

  if thresholds == -1:
    thresholds = int(math.ceil(numModules * 1.0))
  elif thresholds == 0:
    thresholds = numModules
  perModRange = float((90.0 if bumpType == "square" else 60.0) /
                      float(numModules))
  for i in xrange(numModules):
    orientation = (float(i) * perModRange) + (perModRange / 2.0)

    config = {
      "cellsPerAxis": locationModuleWidth,
      "scale": scale,
      "orientation": np.radians(orientation),
      "activationThreshold": 8, # *** to confirm why this is 8
      "initialPermanence": 1.0,
      "connectedPermanence": 0.5,
      "learningThreshold": 8,
      "sampleSize": 10,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.0,
    }

    if bumpType == "square":
      config["cellCoordinateOffsets"] = cellCoordinateOffsets
      config["anchoringMethod"] = anchoringMethod
    elif bumpType == "gaussian":
      config["bumpOverlapMethod"] = "probabilistic"
      config["baselineCellsPerAxis"] = 6
    else:
      raise ValueError("Invalid bumpType", bumpType)

    locationConfigs.append(config)

  l4Overrides = {
    "initialPermanence": 1.0,
    "activationThreshold": thresholds,
    "reducedBasalThreshold": thresholds,
    "minThreshold": numModules,
    "sampleSize": numModules,
    "cellsPerColumn": 16,
  }

  currentNumObjects = numObjects
  numFailuresAllowed = numObjects
  accuracy = None
  allLocationsAreUnique = None # *** is it important to set this to True/False?
  #occurrencesConvergenceLog = []

  #increment = initialIncrement
  #foundUpperBound = False

  print "Testing", numObjects

  column = PIUNCorticalColumn(locationConfigs, L4Overrides=l4Overrides,
                              bumpType=bumpType)


  features = [str(i) for i in xrange(numFeatures)]

  if use_original_objects == True:
    train_objects = generateObjects(numObjects, featuresPerObject, objectWidth,
                              numFeatures, featureDistribution)
    exp = PIUNExperiment_original(column, featureNames=features,
                     numActiveMinicolumns=10,
                     noiseFactor=noiseFactor,
                     moduleNoiseFactor=moduleNoiseFactor)

  else:
    train_features_dic, train_objects = generate_image_objects(numObjects, featuresPerObject, objectWidth,
                              numFeatures, featureDistribution=None, training_bool=True, sanity_check_bool=sanity_check_bool)
    
    train_exp = PIUNExperiment(column, features_dic=train_features_dic,
                         numActiveMinicolumns=10,
                         noiseFactor=noiseFactor,
                         moduleNoiseFactor=moduleNoiseFactor)

    if train_only_bool == True:
      test_features_dic, test_objects = train_features_dic, train_objects
      test_exp = train_exp

    elif train_only_bool == False:
      print("Using seperate test daa-set to evaluate accuracy")
      test_features_dic, test_objects = generate_image_objects(numObjects=100, featuresPerObject=featuresPerObject, objectWidth=objectWidth,
                              numFeatures=5*5*100, featureDistribution=None, training_bool=False)
      test_exp = PIUNExperiment(column, features_dic=test_features_dic,
                     numActiveMinicolumns=10,
                     noiseFactor=noiseFactor,
                     moduleNoiseFactor=moduleNoiseFactor)

  currentLocsUnique = True #***

  for objectDescription in train_objects:
    #print(objectDescription)
    objLocsUnique = train_exp.learnObject(objectDescription)
    currentLocsUnique = currentLocsUnique and objLocsUnique

  numFailures = 0

  # try:
  #   if useTrace:
  #     filename = os.path.join(
  #       SCRIPT_DIR,
  #       "traces/capacity-{}-points-{}-cells-{}-objects-{}-feats.html".format(
  #         len(cellCoordinateOffsets)**2, exp.column.L6aModules[0].numberOfCells(),
  #         currentNumObjects, numFeatures)
  #     )
  #     traceFileOut = io.open(filename, "w", encoding="utf8")
  #     traceHandle = trace(traceFileOut, exp, includeSynapses=False)
  #     print "Logging to", filename

  numIncorrect = 0
  total_sensations = 0

  # Over-write the features dictionary
  #print(train_exp.features)
  train_exp.features = test_features_dic
  #print(train_exp.features)

  objectImages = np.load("first_100_images.npy")
  object_iter = 0
  
  for objectDescription in test_objects:
    numSensationsToInference, incorrect = train_exp.inferObjectWithRandomMovements(
        objectDescription,
        objectImage=objectImages[object_iter],
        trial_iter=ii)
    object_iter += 1
    if numSensationsToInference is None:
      numFailures += 1
      numIncorrect += incorrect
    else:
      print("numSensationsToInference:")
      print(numSensationsToInference)
      total_sensations += numSensationsToInference #Keep a running tally of sensations needed on 
      # successful trials to calculate a mean number of sensations

  # finally:
  #   if useTrace:
  #     traceHandle.__exit__()
  #     traceFileOut.close()

  # if numFailures < numFailuresAllowed:
  currentNumObjects = len(test_objects)
  print("Number of test objects: " + str(currentNumObjects))
  accuracy = 100* float(currentNumObjects - numFailures) / currentNumObjects
  # print(numFailures)
  # print(currentNumObjects)
  # print(numIncorrect)
  errors = 100 * numFailures / float(currentNumObjects)
  false_converging = 100 * numIncorrect / float(currentNumObjects)
  if (currentNumObjects - numFailures) > 0:
    mean_sensations = total_sensations / float(currentNumObjects - numFailures) #Divide by number of successful sensations
    print("Mean sensations:" + str(mean_sensations))
  allLocationsAreUnique = currentLocsUnique



  result = {
    "numObjects": numObjects,
    "accuracy" : accuracy,
    "errors" : errors,
    "false converging": false_converging,
    "allLocationsAreUnique": allLocationsAreUnique,
  }

  print result
  return result

numObjects = 100 #Number of objects to train on 
num_trials = 15
train_only_bool=False
sanity_check_bool = False

use_original_objects = False

all_results = {}
for ii in range(num_trials):

  all_results["trial_" + str(ii)] = basic_object_learning(
                   train_only_bool=train_only_bool,
                   locationModuleWidth=20,
                   bumpType='square',
                   cellCoordinateOffsets=cellCoordinateOffsets,
                   initialIncrement=128,
                   minAccuracy=0.9,
                   capacityResolution=1,
                   capacityPercentageResolution=-1,
                   featuresPerObject=5*5,
                   objectWidth=5,
                   numObjects=numObjects,
                   numFeatures=5*5*numObjects,
                   featureDistribution="AllFeaturesEqual_Replacement",
                   noiseFactor=0,
                   moduleNoiseFactor=0,
                   numModules=40,
                   thresholds=-1,
                   seed1=-1,
                   seed2=-1,
                   anchoringMethod="corners")

  print(all_results)
  with open('all_results.json', 'w') as outfile:
      json.dump(all_results, outfile)


