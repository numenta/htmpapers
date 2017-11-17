# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
This file plots the convergence of L4-L2 as you increase the amount of noise.

"""

import cPickle
import os
from itertools import groupby
from multiprocessing import cpu_count

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy

mpl.rcParams['pdf.fonttype'] = 42

from htmresearch.frameworks.layers.multi_column_convergence_experiment import (
  runExperiment, runExperimentPool
)



def plotConvergenceByObjectMultiColumn(results, objectRange, columnRange):
  """
  Plots the convergence graph: iterations vs number of objects.
  Each curve shows the convergence for a given number of columns.
  """
  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # Convergence[c,o] = how long it took it to converge with f unique features
  # and c columns.

  convergence = numpy.zeros((max(columnRange), max(objectRange) + 1))
  for r in results:
    if r["numColumns"] in columnRange:
      convergence[r["numColumns"] - 1, r["numObjects"]] += r["convergencePoint"]

  convergence /= numTrials

  # print "Average convergence array=", convergence

  ########################################################################
  #
  # Create the plot. x-axis=
  plt.figure()
  plotPath = os.path.join("plots", "convergence_by_object_multicolumn.jpg")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(columnRange)):
    c = columnRange[i]
    print "columns={} objectRange={} convergence={}".format(
      c, objectRange, convergence[c - 1, objectRange])
    if c == 1:
      legendList.append('1 column')
    else:
      legendList.append('{} columns'.format(c))
    plt.plot(objectRange, convergence[c - 1, objectRange],
             color=colorList[i])

  # format
  plt.legend(legendList, loc="upper left", prop={'size': 10})
  plt.xlabel("Number of objects in training set")
  plt.xticks(range(0, max(objectRange) + 1, 10))
  plt.yticks(range(0, int(convergence.max()) + 2))
  plt.ylabel("Average number of touches")
  plt.title("Object recognition with multiple columns (unique features = 5)")

  # save
  plt.savefig(plotPath)
  plt.close()



def plotAccuracyByNoiseLevelAndColumnRange(results, noiseRange, columnRange):
  plt.figure()
  plotPath = os.path.join("plots", "classification_accuracy_by_noiselevelAndColumnNum.jpg")

  # Plot each curve
  accuracyList = []
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for k in range(len(columnRange)):
    c = columnRange[k]
    for i in range(len(noiseRange)):
      accuracyList.append(results[k * len(noiseRange) + i].get('classificationAccuracy'))
    if c == len(columnRange):
      legendList.append('1 column')
    else:
      legendList.append('{} columns'.format(len(columnRange) - k))
    plt.plot(noiseRange, accuracyList, colorList[k])
    accuracyList = []

  # format
  plt.legend(legendList, loc="upper left", prop={'size': 10})
  plt.xlabel("Noise level added")
  plt.ylabel("classification accuracy")
  plt.title("Classification accuracy VS. noise level")

  # save
  plt.savefig(plotPath)
  plt.close()



def plotAccuracyByActivationThreshold(results_by_thresholds, activation_thresholds):
  plt.figure()
  plotPath = os.path.join("plots", "classification_accuracy_by_activationThreshold.jpg")

  plt.plot(activation_thresholds, results_by_thresholds)

  # format
  plt.xlabel("activationThresholdDistal")
  plt.ylabel("classification accuracy")
  plt.title("Classification accuracy VS. activationThresholdDistal")

  # save
  plt.savefig(plotPath)
  plt.close()



def plotFeatureLocationNoise(feature_results, location_results):
  plt.figure()

  feature_results = zip(*feature_results)
  plt.plot(*feature_results, label="Feature Noise")

  location_results = zip(*location_results)
  plt.plot(*location_results, label="Location Noise")

  plt.xlabel("Amount of noise")
  plt.ylabel("Accuracy")
  plt.legend(loc="lower left")

  # save
  plotPath = os.path.join("plots", "feature_location_noise.pdf")
  plt.savefig(plotPath)
  plt.close()



def runFeatureLocationNoiseExperiment():
  """
  Evaluated robustness of a single column network to noise. After the network
  learned a set of objects, we added varying amounts of random noise to the
  sensory and location inputs. The noise affected the active bits in the
  input without changing its overall sparsity (see Materials and Methods).
  Recognition accuracy after 30 touches is plotted as a function of noise
  """
  noise = numpy.arange(0, 0.8, .1)
  objects = [50]
  locations = [10]
  features = [10]
  columns = [1]
  points = 10
  settlingTime = 3
  numTrials = 10

  feature_noise_results = runExperimentPool(
    numObjects=objects,
    numLocations=locations,
    numFeatures=features,
    numColumns=columns,
    featureNoiseRange=noise,
    numWorkers=cpu_count() - 1,
    numPoints=points,
    nTrials=numTrials,
    enableFeedback=[False],
    settlingTime=settlingTime,
    resultsName="feature_noise_results.pkl")

  # with open("feature_noise_results.pkl", "rb") as f:
  #   feature_noise_results = cPickle.load(f)

  # Group Feature results by noise
  feature_noise_key = lambda x: x['featureNoise']
  sorted_results = sorted(feature_noise_results, key=feature_noise_key)
  feature_noise_results = [(k, numpy.mean(next(v)['classificationAccuracy']))
                           for k, v in groupby(sorted_results,
                                               key=feature_noise_key)]

  location_noise_results = runExperimentPool(
    numObjects=objects,
    numLocations=locations,
    numFeatures=features,
    numColumns=columns,
    locationNoiseRange=noise,
    numWorkers=cpu_count() - 1,
    numPoints=points,
    settlingTime=settlingTime,
    nTrials=numTrials,
    enableFeedback=[False],
    resultsName="location_noise_results.pkl")

  # with open("location_noise_results.pkl", "rb") as f:
  #   location_noise_results = cPickle.load(f)

  # Group Location results by noise
  location_noise_key = lambda x: x['locationNoise']
  sorted_results = sorted(location_noise_results, key=location_noise_key)
  location_noise_results = [(k, numpy.mean(next(v)['classificationAccuracy']))
                            for k, v in groupby(sorted_results,
                                                key=location_noise_key)]

  plotFeatureLocationNoise(feature_results=feature_noise_results,
                           location_results=location_noise_results)



def plotSensationsNoise(results):
  plt.figure()

  colorList = ['blue', 'green', 'purple', 'brown', 'fuchsia', 'grey']
  for i, line in enumerate(results):
    plt.plot(line[1], label="Noise level {}".format(line[0]), c=colorList[i])

  plt.xlabel("Number of Sensations")
  plt.ylabel("Accuracy")
  plt.legend(loc="center right")

  # save
  plotPath = os.path.join("plots", "sensation_noise.pdf")
  plt.savefig(plotPath)
  plt.close()



def runSensationsNoiseExperiment():
  """
  Calculcate Recognition accuracy as a function of the number of sensations.
  Colored lines correspond to noise levels in the location input
  """
  noise = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7]
  objects = [100]
  locations = [5000]
  features = [5000]
  columns = [1]
  points = 10
  settlingTime = 3
  numTrials = 10

  results = runExperimentPool(
    numObjects=objects,
    numLocations=locations,
    numFeatures=features,
    numColumns=columns,
    locationNoiseRange=noise,
    numWorkers=cpu_count() - 1,
    numPoints=points,
    settlingTime=settlingTime,
    nTrials=numTrials,
    enableFeedback=[False],
    resultsName="sensation_noise_results.pkl")

  # with open("sensation_noise_results.pkl", "rb") as f:
  #   results = cPickle.load(f)



  # Group results by noise
  noise_key = lambda x: x['locationNoise']
  sorted_results = sorted(results, key=noise_key)
  grouped_results = [(k, numpy.mean([row['classificationPerSensation']
                                     for row in v], axis=0))
                     for k, v in groupby(sorted_results, key=noise_key)]

  plotSensationsNoise(grouped_results)



if __name__ == "__main__":

  # This is how you run a specific experiment in single process mode. Useful
  # for debugging, profiling, etc.
  if False:
    results = runExperiment(
      {
        "numObjects": 100,
        "numPoints": 10,
        "numLocations": 10,
        "numFeatures": 10,
        "numColumns": 1,
        "trialNum": 4,
        "featureNoise": 0.6,
        "plotInferenceStats": False,  # Outputs detailed graphs
        "settlingTime": 3,
        "includeRandomLocation": False,
        "l2Params": {"cellCount": 4096 * 4, "sdrSize": 40 * 2, "activationThresholdDistal": 14}
      }
    )

  # This is for specifically testing how the distal activation threshold affect
  # the classification results
  if True:
    activationThresholdDistalRange = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    results_by_thresholds = []
    for i in range(len(activationThresholdDistalRange)):
      results = runExperiment(
        {
          "numObjects": 100,
          "numPoints": 10,
          "numLocations": 10,
          "numFeatures": 10,
          "numColumns": 1,
          "trialNum": 4,
          "featureNoise": 0.6,
          "plotInferenceStats": False,  # Outputs detailed graphs
          "settlingTime": 3,
          "includeRandomLocation": False,
          "l2Params": {"cellCount": 4096 * 4, "sdrSize": 40 * 2, "activationThresholdDistal": i + 1}
        }
      )
      results_by_thresholds.append(results.get('classificationAccuracy'))

    plotAccuracyByActivationThreshold(results_by_thresholds, activationThresholdDistalRange)

  # Here we want to see how the number of objects affects convergence for
  # multiple columns.
  if False:
    # We run 10 trials for each column number and then analyze results
    numTrials = 1
    columnRange = [1, 2, 3]
    noiseRange = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    featureRange = [10]
    objectRange = [100]

    # Comment this out if you are re-running analysis on already saved results.
    # Very useful for debugging the plots
    runExperimentPool(
      numObjects=objectRange,
      numLocations=[10],
      numFeatures=featureRange,
      numColumns=columnRange,
      numPoints=10,
      featureNoiseRange=noiseRange,
      numWorkers=cpu_count() - 1,
      # numWorkers=1,
      nTrials=numTrials,
      resultsName="object_convergence_noise_results.pkl")

    # Analyze results
    with open("object_convergence_noise_results.pkl", "rb") as f:
      results = cPickle.load(f)
    # print results

    plotAccuracyByNoiseLevelAndColumnRange(results, noiseRange, columnRange)
    # plotConvergenceByObjectMultiColumn(results, objectRange, columnRange)
