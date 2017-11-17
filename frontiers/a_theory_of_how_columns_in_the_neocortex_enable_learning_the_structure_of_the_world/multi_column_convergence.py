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
This file plots the convergence of L4-L2 as you increase the number of columns,
or adjust the confusion between objects.

"""

import cPickle
import os
from multiprocessing import cpu_count

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy

mpl.rcParams['pdf.fonttype'] = 42

from htmresearch.frameworks.layers.multi_column_convergence_experiment import (
  runExperiment, runExperimentPool
)



def plotConvergenceByColumn(results, columnRange, featureRange, numTrials):
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
    convergence[r["numFeatures"] - 1,
                r["numColumns"]] += r["convergencePoint"]
  convergence /= numTrials
  # For each column, print convergence as fct of number of unique features
  for c in range(1, max(columnRange) + 1):
    print c, convergence[:, c]
  # Print everything anyway for debugging
  print "Average convergence array=", convergence
  ########################################################################
  #
  # Create the plot. x-axis=
  plt.figure()
  plotPath = os.path.join("plots", "convergence_by_column.pdf")
  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  for i in range(len(featureRange)):
    f = featureRange[i]
    print columnRange
    print convergence[f - 1, columnRange]
    legendList.append('Unique features={}'.format(f))
    plt.plot(columnRange, convergence[f - 1, columnRange],
             color=colorList[i])
  # format
  plt.legend(legendList, loc="upper right")
  plt.xlabel("Number of columns")
  plt.xticks(columnRange)
  plt.yticks(range(0, int(convergence.max()) + 1))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (multiple columns)")
  # save
  plt.savefig(plotPath)
  plt.close()



def plotConvergenceByObject(results, objectRange, featureRange, numTrials):
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
  plt.figure()
  plotPath = os.path.join("plots", "convergence_by_object_random_location.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    print "features={} objectRange={} convergence={}".format(
      f, objectRange, convergence[f - 1, objectRange])
    legendList.append('Unique features={}'.format(f))
    plt.plot(objectRange, convergence[f - 1, objectRange],
             color=colorList[i])

  # format
  plt.legend(legendList, loc="lower right", prop={'size': 10})
  plt.xlabel("Number of objects in training set")
  plt.xticks(range(0, max(objectRange) + 1, 10))
  plt.yticks(range(0, int(convergence.max()) + 2))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (single column)")

  # save
  plt.savefig(plotPath)
  plt.close()



def plotConvergenceByObjectMultiColumn(results, objectRange, columnRange, numTrials):
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
  plotPath = os.path.join("plots", "convergence_by_object_multicolumn.pdf")

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



def runSingleColumnExperiment():
  """
  Mean number of sensations needed to unambiguously recognize an object with a
  single column network as the set of learned objects increases. We train models
  on varying numbers of objects, from 1 to 100 and plot the average number of
  sensations required to unambiguously recognize a single object.
  """
  # We run 10 trials for each column number and then analyze results
  numTrials = 10
  columnRange = [1]
  featureRange = [5, 10, 20, 30]
  objectRange = [2, 10, 20, 30, 40, 50, 60, 80, 100]

  # Comment this out if you are re-running analysis on already saved results.
  # Very useful for debugging the plots
  runExperimentPool(
    numObjects=objectRange,
    numLocations=[10],
    numFeatures=featureRange,
    numColumns=columnRange,
    numPoints=10,
    nTrials=numTrials,
    numWorkers=cpu_count() - 1,
    resultsName="object_convergence_results.pkl")

  # Analyze results
  with open("object_convergence_results.pkl", "rb") as f:
    results = cPickle.load(f)

  plotConvergenceByObject(results, objectRange, featureRange, numTrials)



def runMultiColumnExperiment():
  """
  Mean number of observations needed to unambiguously recognize an object with
  multi-column networks as the set of columns increases. We train each network
  with 100 objects and plot the average number of sensations required to
  unambiguously recognize an object.
  """
  # We run 10 trials for each column number and then analyze results
  numTrials = 10
  columnRange = [1, 2, 3, 4, 5, 6, 7, 8]
  featureRange = [5, 10, 20, 30]
  objectRange = [100]

  # Comment this out if you are re-running analysis on already saved results.
  # Very useful for debugging the plots
  runExperimentPool(
    numObjects=objectRange,
    numLocations=[10],
    numFeatures=featureRange,
    numColumns=columnRange,
    numPoints=10,
    numWorkers=cpu_count() - 1,
    nTrials=numTrials,
    resultsName="object_convergence_multi_column_results.pkl")

  # Analyze results
  with open("object_convergence_multi_column_results.pkl", "rb") as f:
    results = cPickle.load(f)

  plotConvergenceByColumn(results, columnRange, featureRange, numTrials)



if __name__ == "__main__":

  # This is how you run a specific experiment in single process mode. Useful
  # for debugging, profiling, etc.
  if True:
    results = runExperiment(
      {
        "numObjects": 100,
        "numPoints": 10,
        "numLocations": 10,
        "numFeatures": 10,
        "numColumns": 1,
        "trialNum": 4,
        "featureNoise": 0.0,
        "plotInferenceStats": False,  # Outputs detailed graphs
        "settlingTime": 2,
        "includeRandomLocation": False
      }
    )
