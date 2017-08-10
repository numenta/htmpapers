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
This file contains extra capacity tests not shown in the paper
"""
import os.path

import argparse

from capacity_test import *



def runExperiment1(numObjects=2,
                   sampleSizeRange=(10,),
                   numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  Varying number of pts per objects, two objects
  Try different sample sizes
  """
  objectParams = {'numInputBits': 20,
                  'externalInputSize': 2400,
                  'numFeatures': DEFAULT_NUM_FEATURES,
                  'numLocations': DEFAULT_NUM_LOCATIONS,
                  'uniquePairs': True, }
  l4Params = getL4Params()
  l2Params = getL2Params()

  numInputBits = objectParams['numInputBits']

  l4Params["activationThreshold"] = int(numInputBits * .6)
  l4Params["minThreshold"] = int(numInputBits * .6)
  l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

  for sampleSize in sampleSizeRange:
    print "sampleSize: {}".format(sampleSize)
    l2Params['sampleSizeProximal'] = sampleSize
    expName = "capacity_varying_object_size_synapses_{}".format(sampleSize)
    runCapacityTestVaryingObjectSize(numObjects,
                                     numCorticalColumns,
                                     resultDirName,
                                     expName,
                                     cpuCount,
                                     l2Params,
                                     l4Params,
                                     objectParams=objectParams)

  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying points per object x 2 objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large")

  legendEntries = []

  for sampleSize in sampleSizeRange:
    expName = "capacity_varying_object_size_synapses_{}".format(sampleSize)

    resultFileName = os.path.join(resultDirName, "{}.csv".format(expName))
    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numPointsPerObject", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("# sample size {}".format(sampleSize))

  plt.legend(legendEntries, loc=2)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_size_summary.pdf"
    )
  )



def runExperiment2(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  Try different sample sizes
  """
  sampleSizeRange = (10,)
  numPointsPerObject = 10
  l4Params = getL4Params()
  l2Params = getL2Params()
  objectParams = {'numInputBits': 20,
                  'externalInputSize': 2400,
                  'numFeatures': DEFAULT_NUM_FEATURES,
                  'numLocations': DEFAULT_NUM_LOCATIONS,
                  'uniquePairs': True, }

  for sampleSize in sampleSizeRange:
    print "sampleSize: {}".format(sampleSize)
    l2Params['sampleSizeProximal'] = sampleSize
    expName = "capacity_varying_object_num_synapses_{}".format(sampleSize)

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expName,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(50))

  legendEntries = []
  for sampleSize in sampleSizeRange:
    print "sampleSize: {}".format(sampleSize)
    l2Params['sampleSizeProximal'] = sampleSize
    expName = "capacity_varying_object_num_synapses_{}".format(sampleSize)

    resultFileName = os.path.join(resultDirName, "{}.csv".format(expName))

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("# sample size {}".format(sampleSize))
  ax[0, 0].legend(legendEntries, loc=4, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_summary.pdf"
    )
  )



def runExperiment6(resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  varying size of L2
  calculate capacity by varying number of objects with fixed size
  """

  numPointsPerObject = 10
  numRpts = 5
  numInputBits = 10
  externalInputSize = 2400
  numL4MiniColumns = 150

  l4Params = getL4Params()
  l2Params = getL2Params()

  expParams = [
    {'L2cellCount': 4096, 'L2activeBits': 10, 'w': 10, 'sample': 6, 'thresh': 3,
     'l2Column': 1},
    {'L2cellCount': 4096, 'L2activeBits': 20, 'w': 10, 'sample': 6, 'thresh': 3,
     'l2Column': 1},
    {'L2cellCount': 4096, 'L2activeBits': 40, 'w': 10, 'sample': 6, 'thresh': 3,
     'l2Column': 1},
    {'L2cellCount': 4096, 'L2activeBits': 80, 'w': 10, 'sample': 6, 'thresh': 3,
     'l2Column': 1}
  ]

  for expParam in expParams:
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']
    l2Params['cellCount'] = expParam['L2cellCount']
    l2Params['sdrSize'] = expParam['L2activeBits']
    l2Params['sampleSizeDistal'] = int(l2Params['sdrSize'] / 2)
    l2Params['activationThresholdDistal'] = int(l2Params['sdrSize'] / 2) - 1
    numCorticalColumns = expParam['l2Column']

    l4Params["columnCount"] = numL4MiniColumns
    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {'numInputBits': numInputBits,
                    'externalInputSize': externalInputSize,
                    'numFeatures': DEFAULT_NUM_FEATURES,
                    'numLocations': DEFAULT_NUM_LOCATIONS,
                    'uniquePairs': True, }

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expName = "multiple_column_capacity_varying_object_sdrSize_{}_l2Cells_{}_l2column_{}".format(
      expParam['L2activeBits'], expParam["L2cellCount"], expParam['l2Column'])

    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expName,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle("Varying number of objects", fontsize="x-large")

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expName = "multiple_column_capacity_varying_object_sdrSize_{}_l2Cells_{}_l2column_{}".format(
      expParam['L2activeBits'], expParam["L2cellCount"], expParam['l2Column'])

    resultFileName = os.path.join(resultDirName, "{}.csv".format(expName))

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L2 cells {}/{} #cc {} ".format(
      expParam['L2activeBits'], expParam['L2cellCount'], expParam['l2Column']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_vs_L2_sparsity.pdf"
    )
  )



def runExperiment7(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  Try different numLocations
  """

  numPointsPerObject = 10
  numRpts = 1
  l4Params = getL4Params()
  l2Params = getL2Params()

  l2Params['cellCount'] = 4096
  l2Params['sdrSize'] = 40

  expParams = [
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 16},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 128},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 1000},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 5000},
  ]

  for expParam in expParams:
    l4Params["columnCount"] = expParam['l4Column']
    numInputBits = expParam['w']
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']

    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {
      'numInputBits': numInputBits,
      'externalInputSize': expParam['externalInputSize'],
      'numFeatures': expParam['numFeatures'],
      'numLocations': expParam['numLocations'],
      'uniquePairs': False
    }

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expname = "multiple_column_capacity_varying_object_num_locations_{}_num_features_{}_l4column_{}".format(
      expParam['numLocations'], expParam['numFeatures'], expParam["l4Column"])
    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expname,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expname = "multiple_column_capacity_varying_object_num_locations_{}_num_features_{}_l4column_{}".format(
      expParam['numLocations'], expParam['numFeatures'], expParam["l4Column"])

    result = pd.read_csv(os.path.join(resultDirName, "{}.csv".format(expname)))

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L4 mcs {} locs {} feats {}".format(
      expParam["l4Column"], expParam['numLocations'], expParam['numFeatures']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_locations_num_summary.pdf"
    )
  )



def runExperiment8(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  Try different numFeatures
  """

  numPointsPerObject = 10
  numRpts = 1
  l4Params = getL4Params()
  l2Params = getL2Params()

  l2Params['cellCount'] = 4096
  l2Params['sdrSize'] = 40

  expParams = [
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 15, 'numLocations': 128},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 150, 'numLocations': 128},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 128},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 5000, 'numLocations': 128},
  ]

  for expParam in expParams:
    l4Params["columnCount"] = expParam['l4Column']
    numInputBits = expParam['w']
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']

    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {
      'numInputBits': numInputBits,
      'externalInputSize': expParam['externalInputSize'],
      'numFeatures': expParam['numFeatures'],
      'numLocations': expParam['numLocations'],
      'uniquePairs': False
    }

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expname = "multiple_column_capacity_varying_object_num_locations_{}_num_features_{}_l4column_{}".format(
      expParam['numLocations'], expParam['numFeatures'], expParam["l4Column"])
    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expname,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expname = "multiple_column_capacity_varying_object_num_locations_{}_num_features_{}_l4column_{}".format(
      expParam['numLocations'], expParam['numFeatures'], expParam["l4Column"])

    resultFileName = os.path.join(resultDirName, "{}.csv".format(expname))
    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L4 mcs {} locs {} feats {}".format(
      expParam["l4Column"], expParam['numLocations'], expParam['numFeatures']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_features_num_summary.pdf"
    )
  )



def runExperiment9(resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  varying number of cortical columns, 2d topology.
  """

  numPointsPerObject = 10
  numRpts = 3
  objectNumRange = range(10, 1000, 50)

  l4Params = getL4Params()
  l2Params = getL2Params()

  expParams = [
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 4, 'networkType': "MultipleL4L2Columns"},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 9, 'networkType': "MultipleL4L2Columns"},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 16, 'networkType': "MultipleL4L2Columns"},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 4,
     'networkType': "MultipleL4L2ColumnsWithTopology"},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 9,
     'networkType': "MultipleL4L2ColumnsWithTopology"},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 16,
     'networkType': "MultipleL4L2ColumnsWithTopology"}
  ]

  run_params = []
  for object_num in reversed(objectNumRange):
    for expParam in expParams:
      for rpt in range(numRpts):
        l2Params['sampleSizeProximal'] = expParam['sample']
        l2Params['minThresholdProximal'] = expParam['thresh']

        l4Params["columnCount"] = expParam['l4Column']
        numInputBits = expParam['w']
        numCorticalColumns = expParam['l2Column']
        networkType = expParam['networkType']

        l4Params["activationThreshold"] = int(numInputBits * .6)
        l4Params["minThreshold"] = int(numInputBits * .6)
        l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

        objectParams = {'numInputBits': numInputBits,
                        'externalInputSize': expParam['externalInputSize'],
                        'numFeatures': DEFAULT_NUM_FEATURES,
                        'numLocations': DEFAULT_NUM_LOCATIONS,
                        'uniquePairs': True, }

        print "Experiment Params: "
        pprint(expParam)

        expName = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2column_{}_{}".format(
          expParam['sample'], expParam['thresh'], expParam["l4Column"],
          expParam['l2Column'], expParam["networkType"])

        try:
          os.remove(os.path.join(resultDirName, "{}.csv".format(expName)))
        except OSError:
          pass

        run_params.append((numPointsPerObject,
                           numCorticalColumns,
                           resultDirName,
                           object_num,
                           expName,
                           l2Params,
                           l4Params,
                           objectParams,
                           networkType,
                           rpt))

  pool = multiprocessing.Pool(cpuCount or multiprocessing.cpu_count(),
                              maxtasksperchild=1)
  pool.map(invokeRunCapacityTestWrapper, run_params, chunksize=1)
  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle("Varying number of objects", fontsize="x-large")

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  colormap = plt.get_cmap("jet")
  colors = [colormap(x) for x in np.linspace(0., 1., len(expParam))]

  legendEntries = []
  for expParam in expParams:
    expName = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2column_{}_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"],
      expParam['l2Column'], expParam["networkType"])

    resultFileName = os.path.join(resultDirName, "{}.csv".format(expName))
    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, colors[ploti])
    ploti += 1
    if "Topology" in expParam["networkType"]:
      legendEntries.append("L4 mcs {} #cc {} w/ topology".format(
        expParam['l4Column'], expParam['l2Column']))
    else:
      legendEntries.append("L4 mcs {} #cc {}".format(
        expParam['l4Column'], expParam['l2Column']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "multiple_column_capacity_varying_object_num_column_num_connection_type_summary.pdf"
    )
  )



def runExperiment10(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                    resultDirName=DEFAULT_RESULT_DIR_NAME,
                    plotDirName=DEFAULT_PLOT_DIR_NAME,
                    cpuCount=1):
  """
  Try different L4 network size
  """

  numPointsPerObject = 10
  numRpts = 1
  l4Params = getL4Params()
  l2Params = getL2Params()

  l2Params['cellCount'] = 4096
  l2Params['sdrSize'] = 40

  expParams = []
  expParams.append(
    {'l4Column': 100, 'externalInputSize': 2400, 'w': 10, 'sample': 5,
     'L2cellCount': 2000, 'L2activeBits': 20, 'thresh': 4})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 15, 'sample': 8,
     'L2cellCount': 3000, 'L2activeBits': 30, 'thresh': 6})
  expParams.append(
    {'l4Column': 200, 'externalInputSize': 2400, 'w': 20, 'sample': 10,
     'L2cellCount': 4000, 'L2activeBits': 40, 'thresh': 8})
  expParams.append(
    {'l4Column': 250, 'externalInputSize': 2400, 'w': 25, 'sample': 13,
     'L2cellCount': 5000, 'L2activeBits': 50, 'thresh': 10})

  for expParam in expParams:
    l4Params["columnCount"] = expParam['l4Column']
    numInputBits = expParam['w']

    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']
    l2Params['cellCount'] = expParam['L2cellCount']
    l2Params['sdrSize'] = expParam['L2activeBits']
    l2Params['sampleSizeDistal'] = int(expParam['L2cellCount'] * .5)
    l2Params['activationThresholdDistal'] = int(
      expParam['L2cellCount'] * .5) - 1

    objectParams = {'numInputBits': numInputBits,
                    'externalInputSize': expParam['externalInputSize'],
                    'numFeatures': DEFAULT_NUM_FEATURES,
                    'numLocations': DEFAULT_NUM_LOCATIONS,
                    'uniquePairs': True, }

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expname = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2cell_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"],
      expParam['L2cellCount'])
    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expname,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expname = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2cell_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"],
      expParam['L2cellCount'])

    resultFileName = os.path.join(resultDirName, "{}.csv".format(expname))
    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L4 mcs {} w {} s {} thresh {}".format(
      expParam["l4Column"], expParam['w'], expParam['sample'],
      expParam['thresh']))
  ax[0, 0].legend(legendEntries, loc=4, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_l4l2size_summary.pdf"
    )
  )



def runExperiments(resultDirName, plotDirName, cpuCount):
  #  # Varying number of pts per objects, two objects
  #  runExperiment1(numCorticalColumns=1,
  #                 resultDirName=resultDirName,
  #                 plotDirName=plotDirName,
  #                 cpuCount=cpuCount)
  #
  #  # 10 pts per object, varying number of objects
  #  runExperiment2(numCorticalColumns=1,
  #                 resultDirName=resultDirName,
  #                 plotDirName=plotDirName,
  #                 cpuCount=cpuCount)
  #
  #  # 10 pts per object, varying number of objects, varying L4 size
  #  runExperiment3(numCorticalColumns=1,
  #                 resultDirName=resultDirName,
  #                 plotDirName=plotDirName,
  #                 cpuCount=cpuCount)
  #
  #  # 10 pts per object, varying number of objects and number of columns
  #  runExperiment4(resultDirName=resultDirName,
  #                 plotDirName=plotDirName,
  #                 cpuCount=cpuCount)
  #
  #  # 10 pts per object, varying number of L2 cells
  #  runExperiment5(resultDirName=resultDirName,
  #                 plotDirName=plotDirName,
  #                 cpuCount=cpuCount)
  #
  #  # 10 pts per object, varying sparsity of L2
  #  runExperiment6(resultDirName=resultDirName,
  #                 plotDirName=plotDirName,
  #                 cpuCount=cpuCount)
  #
  #  # 10 pts per object, varying number of location SDRs
  #  runExperiment7(numCorticalColumns=1,
  #                 resultDirName=resultDirName,
  #                 plotDirName=plotDirName,
  #                 cpuCount=cpuCount)
  #
  #  # 10 pts per object, varying number of feature SDRs
  #  runExperiment8(numCorticalColumns=1,
  #                 resultDirName=resultDirName,
  #                 plotDirName=plotDirName,
  #                 cpuCount=cpuCount)
  # 10 pts per object, varying number of objects and number of columns
  runExperiment9(resultDirName=resultDirName,
                 plotDirName=plotDirName,
                 cpuCount=cpuCount)
  #
  # 10 pts per object, varying number of objects, varying L4/L2 size
  # runExperiment10(numCorticalColumns=1,
  #                resultDirName=resultDirName,
  #                plotDirName=plotDirName,
  #                cpuCount=cpuCount)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--resultDirName",
    default=DEFAULT_RESULT_DIR_NAME,
    type=str,
    metavar="DIRECTORY"
  )
  parser.add_argument(
    "--plotDirName",
    default=DEFAULT_PLOT_DIR_NAME,
    type=str,
    metavar="DIRECTORY"
  )
  parser.add_argument(
    "--cpuCount",
    default=None,
    type=int,
    metavar="NUM",
    help="Limit number of cpu cores.  Defaults to `multiprocessing.cpu_count()`"
  )

  opts = parser.parse_args()

  runExperiments(resultDirName=opts.resultDirName,
                 plotDirName=opts.plotDirName,
                 cpuCount=opts.cpuCount)
