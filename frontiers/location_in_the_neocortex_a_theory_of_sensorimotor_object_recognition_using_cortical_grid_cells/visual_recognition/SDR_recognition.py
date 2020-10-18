#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

''' 
This is a basic script to perform object recognition on images using
SDRs generated from a CNN
'''

import math
import os
import random
import time
import json

import numpy as np

from custom_object_creation import (
  PIUNCorticalColumn, PIUNExperiment)
from pre_process_SDRs import generate_image_objects

def object_learning_and_inference(eval_on_training_data_bool,
                 locationModuleWidth,
                 bumpType,
                 cellCoordinateOffsets,
                 cellsPerColumn,
                 columnCount,
                 featuresPerObject,
                 objectWidth,
                 numModules,
                 thresholds,
                 seed1,
                 seed2,
                 anchoringMethod):

  if seed1 != None:
    print("Setting random seed")
    np.random.seed(seed1)

  if seed2 != None:
    print("Setting random seed")
    random.seed(seed2)

  locationConfigs = []
  scale = 40.0

  if thresholds == -1:
    thresholds = int(math.ceil(numModules * 1.0)) # This should be set to 1.0 unless one wants to 
    # simulate a more biologically plausible network where not all cells would be expected to be
    # active.
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
    "activationThreshold": None, # Note some of these are being over-written below, so do not matter
    "initialPermanence": 1.0,
    "connectedPermanence": 0.5,
    "learningThreshold": 8, # As far as I can see, this parameter isn't actually used in apical_tiebreak_temporal_memory.py
    "sampleSize": 10,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.0,
    "cellCoordinateOffsets": cellCoordinateOffsets,
    "anchoringMethod": anchoringMethod
  }

  locationConfigs.append(config)

  l4Overrides = {
    "initialPermanence": 1.0,
    "activationThreshold": thresholds,
    "reducedBasalThreshold": thresholds,
    "minThreshold": numModules,
    "sampleSize": numModules,
    "cellsPerColumn": cellsPerColumn,
    "columnCount": columnCount
  } 

  allLocationsAreUnique = None 

  print("Using " + str(numTrainingObjects) + " training objects.")

  column = PIUNCorticalColumn(locationConfigs, L4Overrides=l4Overrides,
                              bumpType=bumpType)

  train_features_dic, train_objects, object_images = generate_image_objects(data_set, numTrainingObjects, objectWidth, 
      locationModuleWidth, data_set_section='training', sanity_check=sanity_check)
  
  ColumnPlusNet = PIUNExperiment(column, features_dic=train_features_dic) # Note no noise is added

  if eval_on_training_data_bool == True:
    test_features_dic, test_objects, object_images = train_features_dic, train_objects, object_images

  elif eval_on_training_data_bool == False:
    print("Using novel (never trained on) objects to evaluate accuracy")

    # Note that the network trained above
    test_features_dic, test_objects, object_images = generate_image_objects(data_set, numTestingObjects, objectWidth,
                            locationModuleWidth, data_set_section='testing')

  currentLocsUnique = True # Seems to just track that a location hasn't been re-used

  print("\nLearning objects")
  for objectDescription in train_objects:
    print("Learning " + objectDescription["name"])
    start_learn = time.time()

    objLocsUnique = ColumnPlusNet.learnObject(objectDescription)
    currentLocsUnique = currentLocsUnique and objLocsUnique

    print("Time to learn object : " + str(time.time() - start_learn))

  numFailures = 0
  numIncorrect = 0
  numWrongLabel = 0
  numNeverConverged = 0
  total_sensations = 0

  # Over-write the features dictionary for evaluation
  ColumnPlusNet.features = test_features_dic

  print("\nGetting class and non-class targets")
  allClassTargets = []

  for class_iter in range(10):
    print("Targets of " + str(class_iter))
    start_get_targets = time.time()
    allClassTargets.append(ColumnPlusNet.getClassFeatures(objectClass=str(class_iter), featuresPerObject=featuresPerObject))
    print("Time to get class targets : " + str(time.time() - start_get_targets))

  allClassNonTargets = []
  for class_iter in range(10):
    print("Non-targets of " + str(class_iter))
    temp = []
    for non_class_iter in range(10):
      if class_iter != non_class_iter:
        temp.extend(allClassTargets[non_class_iter])

    allClassNonTargets.append(temp)

  # Save model and representations
  # np.save('models/allClassTargets', allClassTargets)
  # np.save('models/allClassNonTargets', allClassNonTargets)
  # np.save('models/ColumnPlusNet', ColumnPlusNet)
  # np.save('models/test_objects', test_objects)
  # np.save('models/object_images', object_images)
  
  object_iter = 0

  for objectDescription in test_objects:
    print("Using class targets from:")
    print(int(objectDescription["name"][0]))
    start_infer = time.time()
    numSensationsToInference, incorrect = ColumnPlusNet.inferObjectWithRandomMovements(
        objectDescription,
        objectImage=object_images[object_iter],
        classTargets=allClassTargets[int(objectDescription["name"][0])],
        nonClassTargets=allClassNonTargets[int(objectDescription["name"][0])],
        trial_iter=ii)
    object_iter += 1
    if numSensationsToInference is None:
      numFailures += 1
      numNeverConverged += incorrect['never_converged']
      numIncorrect += incorrect['false_convergence']
      numWrongLabel += incorrect['wrong_label']
    else:
      print("numSensationsToInference:")
      print(numSensationsToInference)
      total_sensations += numSensationsToInference #Keep a running tally of sensations needed on 
      # successful trials to calculate a mean number of sensations
    print("Time to infer object : " + str(time.time() - start_infer))

  mean_sensations = None

  print("Number of test objects: " + str(numTestingObjects))
  accuracy = 100* float(numTestingObjects - numFailures) / numTestingObjects
  errors = 100 * numFailures / float(numTestingObjects)
  false_converging = 100 * numIncorrect / float(numTestingObjects)
  wrong_labels = 100 * numWrongLabel / float(numTestingObjects)
  never_converged = 100 * numNeverConverged / float(numTestingObjects)
  if (numTestingObjects - numFailures) > 0:
    mean_sensations = total_sensations / float(numTestingObjects - numFailures) #Divide by number of successful sensations
  allLocationsAreUnique = currentLocsUnique



  result = {
    "numTrainingObjects": numTrainingObjects,
    "numTestingObjects" : numTestingObjects,
    "accuracy" : accuracy,
    "total_errors" : errors,
    "wrong_labels" : wrong_labels,
    "false_converging": false_converging,
    "never_converged" : never_converged,
    "mean_sensations" : mean_sensations,
    "allLocationsAreUnique": allLocationsAreUnique,
  }

  print result
  return result



numTrainingObjects = 10
numTestingObjects = 10
num_trials = 1
eval_on_training_data_bool=True
sanity_check = None # Options are 'one_class_training' and 'wrong_label_inference'
data_set = 'mnist'


numOffsets = 2 # Spreads out the activation of grid cells to model uncertainty about location
cellCoordinateOffsets = tuple([i * (0.998 / (numOffsets-1)) + 0.001
                               for i in xrange(numOffsets)])


if os.path.exists('misclassified/') == False:
    try:
        os.mkdir('misclassified/')
    except OSError:
        pass
if os.path.exists('correctly_classified/') == False:
    try:
        os.mkdir('correctly_classified/')
    except OSError:
        pass
if os.path.exists('models/') == False:
    try:
        os.mkdir('models/')
    except OSError:
        pass


all_results = {}
for ii in range(num_trials):

  all_results["trial_" + str(ii)] = object_learning_and_inference(
                   eval_on_training_data_bool=eval_on_training_data_bool,
                   locationModuleWidth=80, # This is the main parameter to increase to improve
                   # representation capacity
                   bumpType='square', # No significance of using this or e.g. Gaussian bump to 
                   # this task
                   cellsPerColumn=16,
                   columnCount=128,
                   cellCoordinateOffsets=cellCoordinateOffsets,
                   featuresPerObject=5*5,
                   objectWidth=5,
                   numModules=40, # 2nd parameter to increase (after locationModuleWidth)
                   thresholds=-1,
                   seed1=123,
                   seed2=123,
                   anchoringMethod="corners") # May see a performance improvement with 
                   # 'discrete', which assumes a noise-less setting; worth trying

  print(all_results)
  with open('all_results.json', 'w') as outfile:
      json.dump(all_results, outfile)


