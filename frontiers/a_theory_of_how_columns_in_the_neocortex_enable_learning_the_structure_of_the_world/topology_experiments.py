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

import os
import numpy
import cPickle
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

from htmresearch.frameworks.layers.multi_column_convergence_experiment import (
  runExperiment
)


def runExperimentPool(numObjects,
                      numLocations,
                      numFeatures,
                      numColumns,
                      networkType=["MultipleL4L2Columns"],
                      longDistanceConnectionsRange = [0.0],
                      numWorkers=7,
                      nTrials=1,
                      pointRange=1,
                      numPoints=10,
                      numInferenceRpts=1,
                      l2Params=None,
                      l4Params=None,
                      resultsName="convergence_results.pkl"):
  """
  Allows you to run a number of experiments using multiple processes.
  For each parameter except numWorkers, pass in a list containing valid values
  for that parameter. The cross product of everything is run, and each
  combination is run nTrials times.

  Returns a list of dict containing detailed results from each experiment.
  Also pickles and saves the results in resultsName for later analysis.

  Example:
    results = runExperimentPool(
                          numObjects=[10],
                          numLocations=[5],
                          numFeatures=[5],
                          numColumns=[2,3,4,5,6],
                          numWorkers=8,
                          nTrials=5)
  """
  # Create function arguments for every possibility
  args = []

  for c in reversed(numColumns):
    for o in reversed(numObjects):
      for l in numLocations:
        for f in numFeatures:
          for n in networkType:
            for p in longDistanceConnectionsRange:
              for t in range(nTrials):
                args.append(
                  {"numObjects": o,
                   "numLocations": l,
                   "numFeatures": f,
                   "numColumns": c,
                   "trialNum": t,
                   "pointRange": pointRange,
                   "numPoints": numPoints,
                   "networkType" : n,
                   "longDistanceConnections" : p,
                   "plotInferenceStats": False,
                   "settlingTime": 3,
                   "numInferenceRpts": numInferenceRpts,
                   "l2Params": l2Params,
                   "l4Params": l4Params
                   }
                )
  print "{} experiments to run, {} workers".format(len(args), numWorkers)
  # Run the pool
  if numWorkers > 1:
    pool = Pool(processes=numWorkers)
    result = pool.map(runExperiment, args)
  else:
    result = []
    for arg in args:
      result.append(runExperiment(arg))

  # print "Full results:"
  # pprint.pprint(result, width=150)

  # Pickle results for later use
  with open(resultsName,"wb") as f:
    cPickle.dump(result,f)

  return result

def plotConvergenceByColumnTopology(results, columnRange, featureRange, networkType, numTrials):
  """
  Plots the convergence graph: iterations vs number of columns.
  Each curve shows the convergence for a given number of unique features.
  """
  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # Convergence[f, c, t] = how long it took it to  converge with f unique
  # features, c columns and topology t.

  convergence = numpy.zeros((max(featureRange), max(columnRange) + 1, len(networkType)))

  networkTypeNames = {}
  for i, topologyType in enumerate(networkType):
    if "Topology" in topologyType:
      networkTypeNames[i] = "Normal"
    else:
      networkTypeNames[i] = "Dense"

  for r in results:
    convergence[r["numFeatures"] - 1, r["numColumns"], networkType.index(r["networkType"])] += r["convergencePoint"]
  convergence /= numTrials

  # For each column, print convergence as fct of number of unique features
  for c in range(1, max(columnRange) + 1):
    for t in range(len(networkType)):
      print c, convergence[:, c, t]

  # Print everything anyway for debugging
  print "Average convergence array=", convergence

  ########################################################################
  #
  # Create the plot. x-axis=
  plt.figure()
  plotPath = os.path.join("plots", "convergence_by_column_topology.pdf")

  # Plot each curve
  legendList = []
  colormap = plt.get_cmap("jet")
  colorList = [colormap(x) for x in numpy.linspace(0., 1.,
      len(featureRange)*len(networkType))]

  for i in range(len(featureRange)):
    for t in range(len(networkType)):
      f = featureRange[i]
      print columnRange
      print convergence[f-1,columnRange, t]
      legendList.append('Unique features={}, topology={}'.format(f, networkTypeNames[t]))
      plt.plot(columnRange, convergence[f-1,columnRange, t],
               color=colorList[i*len(networkType) + t])

  # format
  plt.legend(legendList, loc="upper right")
  plt.xlabel("Number of columns")
  plt.xticks(columnRange)
  plt.yticks(range(0,int(convergence.max())+1))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (multiple columns)")

    # save
  plt.savefig(plotPath)
  plt.close()



def plotConvergenceByDistantConnectionChance(results, featureRange, columnRange, longDistanceConnectionsRange, numTrials):
  """
  Plots the convergence graph: iterations vs number of columns.
  Each curve shows the convergence for a given number of unique features.
  """
  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # Convergence[f, c, t] = how long it took it to  converge with f unique
  # features, c columns and topology t.
  convergence = numpy.zeros((len(featureRange), len(longDistanceConnectionsRange), len(columnRange)))

  for r in results:
      print longDistanceConnectionsRange.index(r["longDistanceConnections"])
      print columnRange.index(r["numColumns"])
      convergence[featureRange.index(r["numFeatures"]),
          longDistanceConnectionsRange.index(r["longDistanceConnections"]),
          columnRange.index(r["numColumns"])] += r["convergencePoint"]

  convergence /= numTrials

  # For each column, print convergence as fct of number of unique features
  for i, c in enumerate(columnRange):
    for j, r in enumerate(longDistanceConnectionsRange):
      print c, r, convergence[:, j, i]

  # Print everything anyway for debugging
  print "Average convergence array=", convergence

  ########################################################################
  #
  # Create the plot. x-axis=
  plt.figure(figsize=(8, 6), dpi=80)
  plotPath = os.path.join("plots", "convergence_by_random_connection_chance.pdf")

  # Plot each curve
  legendList = []
  colormap = plt.get_cmap("jet")
  colorList = [colormap(x) for x in numpy.linspace(0., 1.,
      len(featureRange)*len(longDistanceConnectionsRange))]

  for i, r in enumerate(longDistanceConnectionsRange):
    for j, f in enumerate(featureRange):
      currentColor = i*len(featureRange) + j
      print columnRange
      print convergence[j, i, :]
      legendList.append('Connection_prob = {}, num features = {}'.format(r, f))
      plt.plot(columnRange, convergence[j, i, :], color=colorList[currentColor])

  # format
  plt.legend(legendList, loc = "lower left")
  plt.xlabel("Number of columns")
  plt.xticks(columnRange)
  plt.yticks(range(0,int(convergence.max())+1))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (multiple columns)")

    # save
  plt.show()
  plt.savefig(plotPath)
  plt.close()


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
                    "pointRange": 1,
                    "featureNoise": 0.40,
                    "plotInferenceStats": True,  # Outputs detailed graphs
                    "settlingTime": 3,
                    "includeRandomLocation": False
                  }
    )


  ################
  # These experiments look at the effect of topology

  # Here we want to see how the number of columns affects convergence.
  # This experiment is run using a process pool
  if False:
    columnRange = range(1, 10)
    featureRange = [5]
    objectRange = [100]
    networkType = ["MultipleL4L2Columns", "MultipleL4L2ColumnsWithTopology"]
    numTrials = 10

    # Comment this out if you are re-running analysis on already saved results
    # Very useful for debugging the plots
    runExperimentPool(
      numObjects=objectRange,
      numLocations=[10],
      numFeatures=featureRange,
      numColumns=columnRange,
      networkType=networkType,
      numPoints=10,
      nTrials=numTrials,
      numWorkers=cpu_count() - 1,
      resultsName="column_convergence_results.pkl")

    with open("column_convergence_results.pkl","rb") as f:
      results = cPickle.load(f)

    plotConvergenceByColumnTopology(results, columnRange, featureRange, networkType,
                            numTrials=numTrials)

  # Here we measure the effect of random long-distance connections.
  # We vary the longDistanceConnectionProb parameter,
  if False:
    columnRange = [1,2,3,4,5,6,7,8,9]
    featureRange = [5]
    longDistanceConnectionsRange = [0.0, 0.25, 0.5, 0.9999999]
    objectRange = [100]
    networkType = ["MultipleL4L2ColumnsWithTopology"]
    numTrials = 3

    # Comment this out if you are re-running analysis on already saved results
    # Very useful for debugging the plots
    runExperimentPool(
      numObjects=objectRange,
      numLocations=[10],
      numFeatures=featureRange,
      numColumns=columnRange,
      networkType=networkType,
      longDistanceConnectionsRange = longDistanceConnectionsRange,
      numPoints=10,
      nTrials=numTrials,
      numWorkers=cpu_count() - 1,
      resultsName="random_long_distance_connection_column_convergence_results.pkl")

    with open("random_long_distance_connection_column_convergence_results.pkl","rb") as f:
      results = cPickle.load(f)

    plotConvergenceByDistantConnectionChance(results, featureRange, columnRange,
        longDistanceConnectionsRange, numTrials=numTrials)

