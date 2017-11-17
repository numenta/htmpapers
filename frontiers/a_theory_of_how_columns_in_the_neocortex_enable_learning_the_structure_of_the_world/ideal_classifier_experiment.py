# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
This file runs an idealized observer experiment with feature/location
pairs and compares that to the HTM model.  The ideal observer can also be run
without location information.

Example usage

Train an ideal classifier with/without location

python ideal_classifier_experiment.py --location 1

python ideal_classifier_experiment.py  --location 0
"""

import cPickle
import math
from optparse import OptionParser
import os
import random
from multiprocessing import Pool, cpu_count

import numpy
import numpy as np
import  matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

from htmresearch.frameworks.layers.multi_column_convergence_experiment import (
  runExperimentPool
)

plt.ion()
plt.close('all')
def _getArgs():
  parser = OptionParser(usage="Train classifier")

  parser.add_option("-l",
                    "--location",
                    type=int,
                    default=1,
                    dest="useLocation",
                    help="Whether to use location signal")


  (options, remainder) = parser.parse_args()
  return options, remainder


def findWordInVocabulary(input, wordList):
  findWord = None
  for j in range(wordList.shape[0]):
    numBitDiff = np.sum(np.abs(input - wordList[j, :]))
    if numBitDiff == 0:
      findWord = j
  return findWord



def classifierPredict(testVector, storedVectors):
  """
  Return overlap of the testVector with stored representations for each object.
  """
  numClasses = storedVectors.shape[0]
  output = np.zeros((numClasses,))

  for i in range(numClasses):
    output[i] = np.sum(np.minimum(testVector, storedVectors[i, :]))
  return output


def computeUniquePointsSensed(nCols, nPoints, s):
  """
  If a network with nCols columns senses an object s times, how many
  unique points are actually sensed?  The number is generally <= nCols * s
  because features may be repeated across sensations.
  """
  if nCols == 1:
    return min(s, nPoints)
  elif nCols < nPoints:
    q = float(nCols) / nPoints
    unique = min(int(round(( 1.0 - math.pow(1.0 - q, s)) * nPoints)),
                 nPoints)
    return unique
  else:
    return nPoints


def locateConvergencePoint(stats):
  """
  Walk backwards through stats until you locate the first point that diverges
  from target overlap values.  We need this to handle cases where it might get
  to target values, diverge, and then get back again.  We want the last
  convergence point.
  """
  n = len(stats)
  for i in range(n):
    if stats[n-i-1] < 1:
      return n-i

  # Never differs - converged in one iteration
  return 1



def run_ideal_classifier(args={}):
  """
  Create and train classifier using given parameters.
  """
  numObjects = args.get("numObjects", 10)
  numLocations = args.get("numLocations", 10)
  numFeatures = args.get("numFeatures", 10)
  numPoints = args.get("numPoints", 10)
  trialNum = args.get("trialNum", 42)
  useLocation = args.get("useLocation", 1)
  numColumns = args.get("numColumns", 1)
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=150,
    externalInputSize=2400,
    numCorticalColumns=numColumns,
    numFeatures=numFeatures,
    numLocations=numLocations,
    seed=trialNum
  )
  random.seed(trialNum)

  objects.createRandomObjects(numObjects, numPoints=numPoints,
                              numLocations=numLocations,
                              numFeatures=numFeatures)

  objectSDRs = objects.provideObjectsToLearn()
  objectNames = objectSDRs.keys()
  featureWidth = objects.sensorInputSize
  locationWidth = objects.externalInputSize

  # compute the number of sensations across all objects
  numInputVectors = numPoints * numObjects

  if useLocation:
    inputWidth = featureWidth + locationWidth
  else:
    inputWidth = featureWidth

  # create "training" dataset
  data = np.zeros((numInputVectors, inputWidth))
  label = np.zeros((numInputVectors, numObjects))
  k = 0
  for i in range(numObjects):
    numSensations = len(objectSDRs[objectNames[i]])
    for j in range(numSensations):
      activeBitsFeature = np.array(list(objectSDRs[objectNames[i]][j][0][1]))
      data[k, activeBitsFeature] = 1
      if useLocation:
        activeBitsLocation = np.array(list(objectSDRs[objectNames[i]][j][0][0]))
        data[k, featureWidth + activeBitsLocation] = 1
      label[k, i] = 1
      k += 1

  # enumerate number of distinct "words".
  # Note: this could be done much more easily if we simply use the
  # location/feature pairs that are stored in the object machine.
  wordList = np.zeros((0, inputWidth), dtype='int32')
  featureList = np.zeros((numInputVectors,))
  for i in range(numInputVectors):
    index = findWordInVocabulary(data[i, :], wordList)
    if index is not None:
      featureList[i] = index
    else:
      newWord = np.zeros((1, inputWidth), dtype='int32')
      newWord[0, :] = data[i, :]
      wordList = np.concatenate((wordList, newWord))
      featureList[i] = wordList.shape[0] - 1

  numWords = wordList.shape[0]

  # convert objects to vectorized word representations
  storedObjectRepresentations = np.zeros((numObjects, numWords), dtype=np.int32)
  k = 0
  for i in range(numObjects):
    numSensations = len(objectSDRs[objectNames[i]])
    for j in range(numSensations):
      index = findWordInVocabulary(data[k, :], wordList)
      storedObjectRepresentations[i, index] += 1
      k += 1

  # Cool plot of feature vectors
  # plt.figure()
  # plt.imshow(np.transpose(storedObjectRepresentations))
  # plt.xlabel('Object #')
  # plt.ylabel('Word #')
  # plt.title("Object representations")

  # Create random order of sensations for each object
  objectSensations = []
  for i in range(numObjects):
    senseList = []
    wordIndices = np.where(storedObjectRepresentations[i, :])[0]
    # An object can have multiple instances of a word, in which case we
    # add all of them
    for w in wordIndices:
      senseList.extend(storedObjectRepresentations[i, w]*[w])
    random.shuffle(senseList)
    objectSensations.append(senseList)

  # plot accuracy as a function of number of sensations
  accuracyList = []
  classificationOutcome = np.zeros((numObjects, numPoints+1))
  for sensationNumber in range(1, numPoints+1):
    bowVectorsTest = np.zeros((numObjects, numWords), dtype=np.int32)
    for objectId in range(numObjects):
      # No. sensations for object objectId
      sensations = objectSensations[objectId]
      numPointsToInclude = computeUniquePointsSensed(numColumns,
                                             len(sensations), sensationNumber)
      for j in range(numPointsToInclude):
        index = sensations[j]
        bowVectorsTest[objectId, index] += 1

    # Count the number of correct classifications.
    # A correct classification is where object i is unambiguously recognized.
    numCorrect = 0
    for i in range(numObjects):
      overlaps = classifierPredict(bowVectorsTest[i, :], storedObjectRepresentations)
      bestOverlap = max(overlaps)
      outcome = ( (overlaps[i] == bestOverlap) and
                  len(np.where(overlaps==bestOverlap)[0]) == 1)
      numCorrect += outcome
      classificationOutcome[i, sensationNumber] = outcome
    accuracy = float(numCorrect) / numObjects
    accuracyList.append(accuracy)

  convergencePoint = np.zeros((numObjects, ))
  for i in range(numObjects):
    if np.max(classificationOutcome[i, :])>0:
      convergencePoint[i] = locateConvergencePoint(classificationOutcome[i, :])
    else:
      convergencePoint[i] = 11

  args.update({"accuracy": accuracyList})
  args.update({"numTouches": range(1, 11)})
  args.update({"convergencePoint": np.mean(convergencePoint)})
  args.update({"classificationOutcome": classificationOutcome})
  print "objects={}, features={}, locations={}, distinct words={}, numColumns={}".format(
    numObjects, numFeatures, numLocations, numWords, numColumns),
  print "==> convergencePoint:", args["convergencePoint"]

  return args


def plotConvergenceByObject(results, objectRange, featureRange, numTrials,
                            linestyle='-'):
  """
  Plots the convergence graph: iterations vs number of objects.
  Each curve shows the convergence for a given number of unique features.
  """
  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # Convergence[f,o] = how long it took it to converge with f unique features
  # and o objects.

  convergence = numpy.zeros((max(featureRange), max(objectRange) + 1))
  for r in results:
    if r["numFeatures"] in featureRange:
      convergence[r["numFeatures"] - 1, r["numObjects"]] += r["convergencePoint"]

  convergence /= numTrials

  ########################################################################
  #
  # Create the plot. x-axis=

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    print "features={} objectRange={} convergence={}".format(
      f,objectRange, convergence[f-1,objectRange])
    legendList.append('Unique features={}'.format(f))
    plt.plot(objectRange, convergence[f-1, objectRange],
             color=colorList[i], linestyle=linestyle)

  # format
  plt.legend(legendList, loc="lower right", prop={'size':10})
  plt.xlabel("Number of objects in training set")
  plt.xticks(range(0,max(objectRange)+1,10))
  plt.yticks(range(0,int(convergence.max())+2))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (single column)")




def plotConvergenceByColumn(results, columnRange, featureRange, numTrials, lineStype='-'):
  """
  Plots the convergence graph: iterations vs number of columns.
  Each curve shows the convergence for a given number of unique features.
  """
  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # Convergence[f,c] = how long it took it to  converge with f unique features
  # and c columns.
  convergence = numpy.zeros((max(featureRange), max(columnRange) + 1))
  for r in results:
    if r["numColumns"] > max(columnRange):
      continue
    convergence[r["numFeatures"] - 1,
                r["numColumns"]] += r["convergencePoint"]
  convergence /= numTrials
  # For each column, print convergence as fct of number of unique features
  # for c in range(1, max(columnRange) + 1):
  #   print c, convergence[:, c]
  # Print everything anyway for debugging
  # print "Average convergence array=", convergence
  ########################################################################
  #
  # Create the plot. x-axis=
  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  for i in range(len(featureRange)):
    f = featureRange[i]
    print f, columnRange
    print convergence[f-1,columnRange]
    legendList.append('Unique features={}'.format(f))
    plt.plot(columnRange, convergence[f-1,columnRange],
             color=colorList[i], linestyle=lineStype)
  print
  # format
  plt.legend(legendList, loc="upper right")
  plt.xlabel("Number of columns")
  plt.xticks(columnRange)
  plt.ylim([0, 4])
  plt.yticks(range(0,int(convergence.max())+1))
  plt.ylabel("Average number of sensations")
  plt.title("Number of sensations to recognize one object (multiple columns)")
    # save
  # plotPath = os.path.join("plots", "bow_convergence_by_column.pdf")
  # plt.savefig(plotPath)
  # plt.close()



def run_bow_experiment_single_column(options):
  # Create the objects
  featureRange = [5, 10, 20, 30]
  pointRange = 1
  objectRange = [2, 10, 20, 30, 40, 50, 60, 80, 100]
  numLocations = [10]
  numPoints = 10
  numTrials = 10
  numColumns = [1]
  useLocation = options.useLocation
  args = []
  for c in reversed(numColumns):
    for o in reversed(objectRange):
      for l in numLocations:
        for f in featureRange:
          for t in range(numTrials):
            args.append(
              {"numObjects": o,
               "numLocations": l,
               "numFeatures": f,
               "numColumns": c,
               "trialNum": t,
               "pointRange": pointRange,
               "numPoints": numPoints,
               "useLocation": useLocation
               }

          )

  numWorkers = cpu_count()
  print "{} experiments to run, {} workers".format(len(args), numWorkers)
  print "useLocation: ", useLocation

  pool = Pool(processes=numWorkers)
  result = pool.map(run_ideal_classifier, args)

  resultsName = "bag_of_words_useLocation_{}.pkl".format(useLocation)
  # Pickle results for later use
  with open(resultsName, "wb") as f:
    cPickle.dump(result, f)


  plt.figure()
  with open("object_convergence_results.pkl", "rb") as f:
    results = cPickle.load(f)
  plotConvergenceByObject(results, objectRange, featureRange, linestyle='-')

  with open(resultsName, "rb") as f:
    resultsBOW = cPickle.load(f)
  plotConvergenceByObject(resultsBOW, objectRange, featureRange, numTrials,
                          linestyle='--')
  plotPath = os.path.join("plots",
                          "convergence_by_object_random_location_bow.pdf")
  plt.savefig(plotPath)


def run_multiple_column_experiment():
  """
  Compare the ideal observer against a multi-column sensorimotor network.
  """
  # Create the objects
  featureRange = [5, 10, 20, 30]
  pointRange = 1
  objectRange = [100]
  numLocations = [10]
  numPoints = 10
  numTrials = 10
  columnRange = [1, 2, 3, 4, 5, 6, 7, 8]
  useLocation = 1

  resultsDir = os.path.dirname(os.path.realpath(__file__))

  args = []
  for c in reversed(columnRange):
    for o in reversed(objectRange):
      for l in numLocations:
        for f in featureRange:
          for t in range(numTrials):
            args.append(
              {"numObjects": o,
               "numLocations": l,
               "numFeatures": f,
               "numColumns": c,
               "trialNum": t,
               "pointRange": pointRange,
               "numPoints": numPoints,
               "useLocation": useLocation
               }
            )

  print "Number of experiments:",len(args)
  idealResultsFile = os.path.join(resultsDir,
           "ideal_multi_column_useLocation_{}.pkl".format(useLocation))
  pool = Pool(processes=cpu_count())
  result = pool.map(run_ideal_classifier, args)

  # Pickle results for later use
  with open(idealResultsFile, "wb") as f:
    cPickle.dump(result, f)

  htmResultsFile = os.path.join(resultsDir, "column_convergence_results.pkl")
  runExperimentPool(
    numObjects=objectRange,
    numLocations=[10],
    numFeatures=featureRange,
    numColumns=columnRange,
    numPoints=10,
    nTrials=numTrials,
    numWorkers=cpu_count(),
    resultsName=htmResultsFile)

  with open(htmResultsFile, "rb") as f:
    results = cPickle.load(f)

  with open(idealResultsFile, "rb") as f:
    resultsIdeal = cPickle.load(f)

  plt.figure()
  plotConvergenceByColumn(results, columnRange, featureRange, numTrials)
  plotConvergenceByColumn(resultsIdeal, columnRange, featureRange, numTrials,
                          "--")
  plt.savefig('plots/ideal_observer_multiple_column.pdf')


def single_column_accuracy_comparison():
  """
  Plot accuracy of the ideal observer (with and without locations) as the number
  of sensations increases.
  """
  pointRange = 1
  numTrials = 10
  args = []

  resultsDir = os.path.dirname(os.path.realpath(__file__))

  for t in range(numTrials):
    for useLocation in [0, 1]:
      args.append(
        {"numObjects": 100,
         "numLocations": 10,
         "numFeatures": 10,
         "numColumns": 1,
         "trialNum": t,
         "pointRange": pointRange,
         "numPoints": 10,
         "useLocation": useLocation
         }
      )

  print "{} experiments to run, {} workers".format(len(args), cpu_count())

  idealResultsFile = os.path.join(resultsDir, "ideal_model_result.pkl")

  # Run all experiments and pickle results for later use
  pool = Pool(processes=cpu_count())
  resultsIdeal = pool.map(run_ideal_classifier, args)

  with open(idealResultsFile, "wb") as f:
    cPickle.dump(resultsIdeal, f)

  # run HTM network
  columnRange = [1]
  objectRange = [100]
  numAmbiguousLocationsRange = [0]
  htmResultsFile = os.path.join(resultsDir, "single_column_convergence_results.pkl")
  runExperimentPool(
    numObjects=objectRange,
    numLocations=[10],
    numFeatures=[10],
    numColumns=columnRange,
    numPoints=10,
    settlingTime=1,
    nTrials=numTrials,
    numWorkers=cpu_count(),
    ambiguousLocationsRange=numAmbiguousLocationsRange,
    resultsName=htmResultsFile)

  # Read results from pickle files
  with open(idealResultsFile, "rb") as f:
    resultsIdeal = cPickle.load(f)
  with open(htmResultsFile, "rb") as f:
    resultsModel = cPickle.load(f)

  # plot accuracy across sensations
  accuracyIdeal = 0
  accuracyBOF = 0
  for r in resultsIdeal:
    if r["useLocation"]:
      accuracyIdeal += np.array(r['accuracy'])
    else:
      accuracyBOF += np.array(r['accuracy'])

  accuracyIdeal /= len(resultsIdeal) / 2
  accuracyBOF /= len(resultsIdeal) / 2

  numTouches = len(accuracyIdeal)
  accuracyModel = 0
  for r in resultsModel:
    accuracyModel += np.array(r['classificationPerSensation'])
  accuracyModel /= len(resultsModel)

  plt.figure()
  plt.plot(np.arange(numTouches)+1, accuracyIdeal, '-o', label='Ideal observer (with location')
  plt.plot(np.arange(numTouches) + 1, accuracyBOF, '-s', label='Ideal observer (no location)')
  plt.plot(np.arange(numTouches)+1, accuracyModel, '-^', label='Sensorimotor network')
  plt.xlabel("Number of sensations")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.savefig('plots/ideal_observer_comparison_single_column.pdf')


def runTests():
  # object  # 100 feature #  10 location # 10 distinct words # 100 numColumns 1
  # convergencePoint: 2.29
  run_ideal_classifier(
    {
      "numObjects": 100,
      "numPoints": 10,
      "numColumns": 1,
    }
  )

  # object  # 53 feature #  13 location # 10 distinct words # 126 numColumns 1
  # convergencePoint: 1.88679245283
  run_ideal_classifier(
    {
      "numObjects": 53,
      "numPoints": 9,
      "numFeatures": 13,
      "numColumns": 1,
    }
  )

  # object  # 101 feature #  11 location # 10 distinct words # 110 numColumns 1
  # convergencePoint: 2.27722772277
  run_ideal_classifier(
    {
      "numObjects": 101,
      "numPoints": 10,
      "numFeatures": 11,
      "numColumns": 1,
    }
  )

  # Should be faster than above
  # convergencePoint: 2.27722772277
  run_ideal_classifier(
    {
      "numObjects": 101,
      "numPoints": 10,
      "numFeatures": 11,
      "numColumns": 2,
    }
  )

if __name__ == "__main__":
  (options, args) = _getArgs()

  # run_bow_experiment_single_column(options)

  # run_multiple_column_experiment()

  single_column_accuracy_comparison()

  # runTests()

