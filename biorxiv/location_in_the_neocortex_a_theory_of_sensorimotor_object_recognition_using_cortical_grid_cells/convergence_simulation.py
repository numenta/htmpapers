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
Convergence simulations for abstract objects.
"""

import argparse
import collections
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
from htmresearch.frameworks.location.path_integration_union_narrowing import (
  PIUNCorticalColumn, PIUNExperiment, PIUNExperimentMonitor)
from htmresearch.frameworks.location.two_layer_tracing import (
  PIUNVisualizer as trace, PIUNLogger as rawTrace)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))



class PIUNCellActivityTracer(PIUNExperimentMonitor):
  def __init__(self, exp):
    self.exp = exp
    self.locationLayerTimelineByObject = {}
    self.inferredStepByObject = {}
    self.currentObjectName = None

  def afterLocationAnchor(self, **kwargs):
    moduleStates = [{"activeCells": module.activeCells.tolist()}
                    for module in self.exp.column.L6aModules]

    if self.exp.column.bumpType == "gaussian":
      for iModule, module in enumerate(self.exp.column.L6aModules):
        moduleStates[iModule]["bumps"] = module.bumpPhases.T.tolist()

    self.locationLayerTimelineByObject[self.currentObjectName].append(
      moduleStates)

  def beforeInferObject(self, obj):
    self.currentObjectName = obj["name"]
    self.locationLayerTimelineByObject[obj["name"]] = []

  def afterInferObject(self, obj, inferredStep):
    self.inferredStepByObject[obj["name"]] = inferredStep



def doExperiment(locationModuleWidth,
                 bumpType,
                 cellCoordinateOffsets,
                 numObjects,
                 featuresPerObject,
                 objectWidth,
                 numFeatures,
                 featureDistribution,
                 useTrace,
                 useRawTrace,
                 logCellActivity,
                 logNumFeatureOccurrences,
                 noiseFactor,
                 moduleNoiseFactor,
                 numModules,
                 numSensations,
                 thresholds,
                 seed1,
                 seed2,
                 anchoringMethod):
  """
  Learn a set of objects. Then try to recognize each object. Output an
  interactive visualization.

  @param locationModuleWidth (int)
  The cell dimensions of each module

  @param cellCoordinateOffsets (sequence)
  The "cellCoordinateOffsets" parameter for each module
  """
  if not os.path.exists("traces"):
    os.makedirs("traces")

  if seed1 != -1:
    np.random.seed(seed1)

  if seed2 != -1:
    random.seed(seed2)

  features = [str(i) for i in xrange(numFeatures)]
  objects = generateObjects(numObjects, featuresPerObject, objectWidth,
                            numFeatures, featureDistribution)

  if logNumFeatureOccurrences:
    featureOccurrences = collections.Counter(feat["name"]
                                             for obj in objects
                                             for feat in obj["features"])
    occurrencesConvergenceLog = []

  locationConfigs = []
  scale = 40.0

  if thresholds == -1:
    thresholds = int(math.ceil(numModules*0.8))
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
      "activationThreshold": 8,
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

  column = PIUNCorticalColumn(locationConfigs, L4Overrides=l4Overrides,
                              bumpType=bumpType)
  exp = PIUNExperiment(column, featureNames=features,
                       numActiveMinicolumns=10,
                       noiseFactor=noiseFactor,
                       moduleNoiseFactor=moduleNoiseFactor)

  for objectDescription in objects:
    exp.learnObject(objectDescription)

  convergence = collections.defaultdict(int)

  try:
    if useTrace:
      filename = os.path.join(
        SCRIPT_DIR,
        "traces/{}-points-{}-cells-{}-objects-{}-feats.html".format(
          len(cellCoordinateOffsets)**2, exp.column.L6aModules[0].numberOfCells(),
          numObjects, numFeatures)
      )
      traceFileOut = io.open(filename, "w", encoding="utf8")
      traceHandle = trace(traceFileOut, exp, includeSynapses=True)
      print "Logging to", filename

    if useRawTrace:
      rawFilename = os.path.join(
        SCRIPT_DIR,
        "traces/{}-points-{}-cells-{}-objects-{}-feats.trace".format(
          len(cellCoordinateOffsets)**2, exp.column.L6aModules[0].numberOfCells(),
          numObjects, numFeatures)
      )
      rawTraceFileOut = open(rawFilename, "w")
      rawTraceHandle = rawTrace(rawTraceFileOut, exp, includeSynapses=False)
      print "Logging to", rawFilename

    if logCellActivity:
      cellActivityTracer = PIUNCellActivityTracer(exp)
      exp.addMonitor(cellActivityTracer)

    for objectDescription in objects:

      numSensationsToInference = exp.inferObjectWithRandomMovements(
        objectDescription, numSensations)

      if logNumFeatureOccurrences:
        objectFeatureOccurrences = sorted(featureOccurrences[feat["name"]]
                                          for feat in objectDescription["features"])
        occurrencesConvergenceLog.append(
          (objectFeatureOccurrences, numSensationsToInference))
      convergence[numSensationsToInference] += 1
      if numSensationsToInference is None:
        print 'Failed to infer object "{}"'.format(objectDescription["name"])
  finally:
    if useTrace:
      traceHandle.__exit__()
      traceFileOut.close()

    if useRawTrace:
      rawTraceHandle.__exit__()
      rawTraceFileOut.close()

  for step, num in sorted(convergence.iteritems()):
    print "{}: {}".format(step, num)

  result = {
    "convergence": convergence,
  }

  if bumpType == "gaussian":
    result["bumpSigma"] = column.L6aModules[0].bumpSigma

  if logCellActivity:
    result["locationLayerTimelineByObject"] = (
      cellActivityTracer.locationLayerTimelineByObject)
    result["inferredStepByObject"] = cellActivityTracer.inferredStepByObject

  if logNumFeatureOccurrences:
    result["occurrencesConvergenceLog"] = occurrencesConvergenceLog

  return result


def experimentWrapper(args):
  return doExperiment(**args)


def runMultiprocessNoiseExperiment(resultName, repeat, numWorkers,
                                   appendResults, **kwargs):
  """
  :param kwargs: Pass lists to distribute as lists, lists that should be passed intact as tuples.
  :return: results, in the format [(arguments, results)].  Also saved to json at resultName, in the same format.
  """
  experiments = [{}]
  for key, values in kwargs.items():
    if type(values) is list:
      newExperiments = []
      for experiment in experiments:
        for val in values:
          newExperiment = copy(experiment)
          newExperiment[key] = val
          newExperiments.append(newExperiment)
      experiments = newExperiments
    else:
      [a.__setitem__(key, values) for a in experiments]

  newExperiments = []
  for experiment in experiments:
    for _ in xrange(repeat):
      newExperiments.append(copy(experiment))
  experiments = newExperiments

  return runExperiments(experiments, resultName, numWorkers, appendResults)


def runExperiments(experiments, resultName, numWorkers=-1, appendResults=False):
  if numWorkers == -1:
    numWorkers = cpu_count()

  if numWorkers > 1:
    pool = Pool(processes=numWorkers)
    rs = pool.map_async(experimentWrapper, experiments, chunksize=1)
    while not rs.ready():
      remaining = rs._number_left
      pctDone = 100.0 - (100.0*remaining) / len(experiments)
      print "    =>", remaining, "experiments remaining, percent complete=",pctDone
      time.sleep(5)
    pool.close()  # No more work
    pool.join()
    result = rs.get()
  else:
    result = []
    for arg in experiments:
      result.append(doExperiment(**arg))

  # Save results for later use
  results = [(arg,res) for arg, res in zip(experiments, result)]

  if appendResults:
    try:
      with open(os.path.join(SCRIPT_DIR, resultName), "r") as f:
        existingResults = json.load(f)
        results = existingResults + results
    except IOError:
      pass

  with open(os.path.join(SCRIPT_DIR, resultName),"wb") as f:
    print "Writing results to {}".format(resultName)
    json.dump(results,f)

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--numObjects", type=int, nargs="+", required=True)
  parser.add_argument("--numUniqueFeatures", type=int, nargs="+", required=True)
  parser.add_argument("--locationModuleWidth", type=int, nargs="+", required=True)
  parser.add_argument("--bumpType", type=str, nargs="+", default="gaussian",
                      help="Set to 'square' or 'gaussian'")
  parser.add_argument("--coordinateOffsetWidth", type=int, default=2)
  parser.add_argument("--noiseFactor", type=float, nargs="+", required=False, default = 0)
  parser.add_argument("--moduleNoiseFactor", type=float, nargs="+", required=False, default=0)
  parser.add_argument("--useTrace", action="store_true")
  parser.add_argument("--useRawTrace", action="store_true")
  parser.add_argument("--logCellActivity", action="store_true")
  parser.add_argument("--logNumFeatureOccurrences", action="store_true")
  parser.add_argument("--numModules", type=int, nargs="+", default=[20])
  parser.add_argument("--numSensations", type=int, default=-1)
  parser.add_argument("--seed1", type=int, default=-1)
  parser.add_argument("--seed2", type=int, default=-1)
  parser.add_argument(
    "--thresholds", type=int, default=-1,
    help=(
      "The TM prediction threshold. Defaults to ceil(numModules*0.8)."
      "Set to 0 for the threshold to match the number of modules."))
  parser.add_argument("--featuresPerObject", type=int, nargs="+", default=10)
  parser.add_argument("--featureDistribution", type = str, nargs="+",
                      default="AllFeaturesEqual_Replacement")
  parser.add_argument("--anchoringMethod", type = str, default="corners")
  parser.add_argument("--resultName", type = str, default="results.json")
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument("--appendResults", action="store_true")
  parser.add_argument("--numWorkers", type=int, default=cpu_count())

  args = parser.parse_args()

  numOffsets = args.coordinateOffsetWidth
  cellCoordinateOffsets = tuple([i * (0.998 / (numOffsets-1)) + 0.001 for i in xrange(numOffsets)])

  if "both" in args.anchoringMethod:
    args.anchoringMethod = ["narrowing", "corners"]

  runMultiprocessNoiseExperiment(
    args.resultName, args.repeat, args.numWorkers, args.appendResults,
    locationModuleWidth=args.locationModuleWidth,
    bumpType=args.bumpType,
    cellCoordinateOffsets=cellCoordinateOffsets,
    numObjects=args.numObjects,
    featuresPerObject=args.featuresPerObject,
    featureDistribution=args.featureDistribution,
    objectWidth=4,
    numFeatures=args.numUniqueFeatures,
    useTrace=args.useTrace,
    useRawTrace=args.useRawTrace,
    logCellActivity=args.logCellActivity,
    logNumFeatureOccurrences=args.logNumFeatureOccurrences,
    noiseFactor=args.noiseFactor,
    moduleNoiseFactor=args.moduleNoiseFactor,
    numModules=args.numModules,
    numSensations=(args.numSensations if args.numSensations != -1
                   else None),
    thresholds=args.thresholds,
    anchoringMethod=args.anchoringMethod,
    seed1=args.seed1,
    seed2=args.seed2,
  )
