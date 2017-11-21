#!/usr/bin/env python
# ----------------------------------------------------------------------
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

import os
from optparse import OptionParser
import pprint
import pickle
from tabulate import tabulate

import nupic.math.topology as topology

from nupic.algorithms.spatial_pooler import SpatialPooler as PYSpatialPooler
from htmresearch.algorithms.faulty_spatial_pooler import FaultySpatialPooler
from htmresearch.frameworks.sp_paper.sp_metrics import (
  calculateEntropy, inspectSpatialPoolerStats,
  classificationAccuracyVsNoise, getRFCenters, calculateOverlapCurve,
  calculateStability, calculateInputSpaceCoverage, plotExampleInputOutput,
  reconstructionError, witnessError
)
from htmresearch.support.spatial_pooler_monitor_mixin import (
  SpatialPoolerMonitorMixin)

from htmresearch.support.sp_paper_utils import *

class MonitoredSpatialPooler(SpatialPoolerMonitorMixin,
                             PYSpatialPooler): pass

class MonitoredFaultySpatialPooler(SpatialPoolerMonitorMixin,
                                   FaultySpatialPooler): pass


from nupic.bindings.algorithms import SpatialPooler as CPPSpatialPooler
from nupic.bindings.math import GetNTAReal

from nupic.math.topology import indexFromCoordinates

from htmresearch.support.generate_sdr_dataset import SDRDataSet

realDType = GetNTAReal()
uintType = "uint32"
plt.ion()
mpl.rcParams['pdf.fonttype'] = 42



def createSpatialPooler(spatialImp, spatialPoolerParameters):
  if spatialImp == 'py':
    sp = PYSpatialPooler(**spatialPoolerParameters)
  elif spatialImp == 'cpp':
    sp = CPPSpatialPooler(**spatialPoolerParameters)
  elif spatialImp == 'monitored_sp':
    sp = MonitoredSpatialPooler(**spatialPoolerParameters)
  elif spatialImp == "faulty_sp":
    sp = FaultySpatialPooler(**spatialPoolerParameters)
  elif spatialImp == "monitored_faulty_sp":
    sp = MonitoredFaultySpatialPooler(**spatialPoolerParameters)

  else:
    raise RuntimeError("Invalide spatialImp")
  return sp



def getSpatialPoolerParams(params, expConfig):
  if expConfig.topology == 1:
    if expConfig.boosting > 0:
      from model_params.sp_params import \
        spParamTopologyWithBoosting as spatialPoolerParameters
    else:
      from model_params.sp_params import \
        spParamTopologyNoBoosting as spatialPoolerParameters

    if expConfig.dataSet in ['randomCross']:
      from model_params.sp_params import \
        spParamTopologyWithBoostingCross as spatialPoolerParameters

    spatialPoolerParameters['inputDimensions'] = (params['nX'], params['nY'])
  elif expConfig.topology == 0:
    if expConfig.boosting == 0:
      from model_params.sp_params import \
        spParamNoBoosting as spatialPoolerParameters
    else:
      from model_params.sp_params import \
        spParamWithBoosting as spatialPoolerParameters

    if expConfig.dataSet in ['mnist']:
      spatialPoolerParameters['inputDimensions'] = (1024, 1)
  print
  print "Spatial Pooler Parameters: "
  pprint.pprint(spatialPoolerParameters)
  return spatialPoolerParameters



def plotReceptiveFields2D(sp, Nx, Ny, seed=42, nrows=4, ncols=4):
  inputSize = Nx * Ny
  numColumns = np.product(sp.getColumnDimensions())

  fig, ax = plt.subplots(nrows, ncols)
  np.random.seed(seed)
  for r in range(nrows):
    for c in range(ncols):
      colID = np.random.randint(numColumns)
      connectedSynapses = np.zeros((inputSize,), dtype=uintType)
      sp.getConnectedSynapses(colID, connectedSynapses)

      potentialSyns = np.zeros((inputSize,), dtype=uintType)
      sp.getPotential(colID, potentialSyns)

      receptiveField = np.zeros((inputSize,), dtype=uintType)
      receptiveField[potentialSyns > 0] = 1
      receptiveField[connectedSynapses > 0] = 5
      receptiveField = np.reshape(receptiveField, (Nx, Ny))

      ax[r, c].imshow(receptiveField, interpolation="nearest", cmap='gray')
      ax[r, c].set_xticks([])
      ax[r, c].set_yticks([])
      ax[r, c].set_title('col {}'.format(colID))
  return fig



def _getArgs():
  parser = OptionParser(usage="Train HTM Spatial Pooler")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='randomSDR',
                    dest="dataSet",
                    help="DataSet Name, choose from sparse, correlated-input"
                         "bar, cross, image")

  parser.add_option("-b",
                    "--boosting",
                    type=int,
                    default=0,
                    dest="boosting",
                    help="Whether to use boosting")

  parser.add_option("-e",
                    "--numEpochs",
                    type=int,
                    default=100,
                    dest="numEpochs",
                    help="number of epochs")

  parser.add_option("-c",
                    "--runClassification",
                    type=int,
                    default=0,
                    dest="classification",
                    help="Whether to run classification experiment")

  parser.add_option("--spatialImp",
                    type=str,
                    default="cpp",
                    dest="spatialImp",
                    help="spatial pooler implementations: py, c++, or "
                         "monitored_sp")

  parser.add_option("-t", "--topology",
                    type=int,
                    default=0,
                    dest="topology",
                    help="0: no topology, global inhibition, 1: with topology")

  parser.add_option("--trackOverlapCurve",
                    type=int,
                    default=0,
                    dest="trackOverlapCurve",
                    help="whether to track overlap curve during learning")

  parser.add_option("--checkRFCenters",
                    type=int,
                    default=0,
                    dest="checkRFCenters",
                    help="whether to track RF cneters")

  parser.add_option("--checkTestInput",
                    type=int,
                    default=0,
                    dest="checkTestInput",
                    help="whether to check response to test inputs")

  parser.add_option("--checkInputSpaceCoverage",
                    type=int,
                    default=0,
                    dest="checkInputSpaceCoverage",
                    help="whether to check coverage of input space")

  parser.add_option("--saveBoostFactors",
                    type=int,
                    default=0,
                    dest="saveBoostFactors",
                    help="save boost factors for future inspection")

  parser.add_option("--changeDataSetContinuously",
                    type=int,
                    default=0,
                    dest="changeDataSetContinuously",
                    help="whether to change data set at every epoch")

  parser.add_option("--showExampleRFs",
                    type=int,
                    default=0,
                    dest="showExampleRFs",
                    help="whether to show example RFs over training")

  parser.add_option("--changeDataSetAt",
                    type=int,
                    default=-1,
                    dest="changeDataSetAt",
                    help="regenerate dataset at this iteration")

  parser.add_option("--killCellsAt",
                    type=int,
                    default=-1,
                    dest="killCellsAt",
                    help="kill a fraction of sp cells at this iteration")

  parser.add_option("--killCellPrct",
                    type=float,
                    default=0.0,
                    dest="killCellPrct",
                    help="the fraction of sp cells that will be removed")

  parser.add_option("--killInputsAfter",
                    type=int,
                    default=-1,
                    dest="killInputsAfter",
                    help="kill a fraction of inputs")

  parser.add_option("--name",
                    type=str,
                    default='defaultName',
                    dest="expName",
                    help="the fraction of sp cells that will be removed")

  parser.add_option("--seed",
                    type=str,
                    default=41,
                    dest="seed",
                    help="random seed for SP and dataset")

  (options, remainder) = parser.parse_args()
  print options
  return options, remainder



def updatePotentialRadius(sp, newPotentialRadius):
  """
  Change the potential radius for all columns
  :return:
  """
  oldPotentialRadius = sp._potentialRadius
  sp._potentialRadius = newPotentialRadius
  numColumns = np.prod(sp.getColumnDimensions())
  for columnIndex in xrange(numColumns):
    potential = sp._mapPotential(columnIndex)
    sp._potentialPools.replace(columnIndex, potential.nonzero()[0])

  sp._updateInhibitionRadius()



def initializeSPConnections(sp, potentialRaidus=10, initConnectionRadius=5):
  numColumns = np.prod(sp.getColumnDimensions())

  updatePotentialRadius(sp, newPotentialRadius=initConnectionRadius)
  for columnIndex in xrange(numColumns):
    potential = sp._mapPotential(columnIndex)
    sp._potentialPools.replace(columnIndex, potential.nonzero()[0])
    perm = sp._initPermanence(potential, 0.5)
    sp._updatePermanencesForColumn(perm, columnIndex, raisePerm=True)

  updatePotentialRadius(sp, newPotentialRadius=potentialRaidus)



def getSDRDataSetParams(inputVectorType, seed):
  if inputVectorType == 'randomBarSets':
    params = {'dataType': 'randomBarSets',
              'numInputVectors': 100,
              'nX': 32,
              'nY': 32,
              'barHalfLength': 6,
              'numBarsPerInput': 6,
              'seed': seed}
  elif inputVectorType == 'randomBarPairs':
    params = {'dataType': 'randomBarSets',
              'numInputVectors': 1000,
              'nX': 32,
              'nY': 32,
              'barHalfLength': 3,
              'numBarsPerInput': 2,
              'seed': seed}
  elif inputVectorType == 'randomCross':
    params = {'dataType': 'randomCross',
              'numInputVectors': 1000,
              'nX': 32,
              'nY': 32,
              'barHalfLength': 3,
              'numCrossPerInput': 6,
              'seed': seed}
  elif inputVectorType == 'randomSDRVaryingSparsity':
    params = {'dataType': 'randomSDRVaryingSparsity',
              'numInputVectors': 100,
              'inputSize': 1024,
              'nX': 32,
              'nY': 32,
              'minSparsity': 0.02,
              'maxSparsity': 0.2,
              'seed': seed}
  elif inputVectorType == 'randomSDR':
    params = {'dataType': 'randomSDR',
              'numInputVectors': 200,
              'inputSize': 1024,
              'nX': 32,
              'nY': 32,
              'numActiveInputBits': 20,
              'seed': seed}
  elif inputVectorType == 'mnist':
    params = {'dataType': 'mnist',
              'numInputVectors': 10000,
              'inputSize': 1024,
              'nX': 32,
              'nY': 32,
              'seed': seed}
  else:
    raise ValueError('unknown data type')
  print
  print "dataset parameters"
  pprint.pprint(params)
  return params



def getExperimentName(expConfig):
  if expConfig.expName == 'defaultName':
    expName = "dataType_{}_boosting_{}".format(
      expConfig.dataSet, expConfig.boosting)
  else:
    expName = expConfig.expName
  expName = expName + '_seed_{}'.format(expConfig.seed)
  return expName



def createDirectories(expName):
  def createDirectoryIfNotExist(directory):
    if not os.path.exists(directory):
      os.makedirs(directory)

  auxDirectoryList = [
    'figures/exampleInputs/',
    'results/input_output_overlap/{}/'.format(expName),
    'results/classification/{}/'.format(expName),
    'figures/RFcenters/{}/'.format(expName),
    'results/RFcenters/{}/'.format(expName),
    'results/InputCoverage/{}/'.format(expName),
    'figures/InputCoverage/{}/'.format(expName),
    'figures/ResponseToTestInputs/{}/'.format(expName),
    'results/boostFactors/{}/'.format(expName),
    'figures/exampleRFs/{}/'.format(expName),
    'results/traces/{}/'.format(expName),
  ]
  for dir in auxDirectoryList:
    createDirectoryIfNotExist(dir)



def runSPexperiments(expConfig):
  inputVectorType = expConfig.dataSet
  params = getSDRDataSetParams(expConfig.dataSet, int(expConfig.seed))
  expName = getExperimentName(expConfig)
  createDirectories(expName)

  sdrData = SDRDataSet(params)
  inputVectors = sdrData.getInputVectors()
  numInputVector, inputSize = inputVectors.shape

  plt.figure()
  plt.imshow(np.reshape(inputVectors[2], (params['nX'], params['nY'])),
             interpolation='nearest', cmap='gray')
  plt.savefig('figures/exampleInputs/{}'.format(expName))

  print
  print "Runnning experiment: {}".format(expName)
  print "Training Data Size {} Dimensions {}".format(numInputVector, inputSize)
  print "Number of epochs",expConfig.numEpochs

  spParams = getSpatialPoolerParams(params, expConfig)
  sp = createSpatialPooler(expConfig.spatialImp, spParams)

  if expConfig.topology == 1 and  expConfig.spatialImp in ['faulty_sp', 'py']:
    initializeSPConnections(sp, potentialRaidus=10, initConnectionRadius=5)

  numColumns = np.prod(sp.getColumnDimensions())

  numTestInputs = min(int(numInputVector * 0.5), 1000)
  testInputs = inputVectors[:numTestInputs, :]

  connectedCounts = np.zeros((numColumns,), dtype=uintType)
  boostFactors = np.zeros((numColumns,), dtype=realDType)
  activeDutyCycle = np.zeros((numColumns,), dtype=realDType)

  metrics = {'numConnectedSyn': [],
             'numNewSyn': [],
             'numRemoveSyn': [],
             'stability': [],
             'entropy': [],
             'maxEntropy': [],
             'sparsity': [],
             'noiseRobustness': [],
             'classification': [],
             'meanBoostFactor': [],
             'reconstructionError': [],
             'witnessError': []}

  connectedSyns = getConnectedSyns(sp)

  print "Running discrimination test before training"
  runDiscriminationTest(sp, inputVectors, numPairs=1000)

  activeColumnsCurrentEpoch, dum = runSPOnBatch(sp, testInputs, learn=False)

  inspectSpatialPoolerStats(sp, inputVectors, expName + "beforeTraining")

  checkPoints = [0, expConfig.changeDataSetAt - 1,
                 expConfig.changeDataSetAt, expConfig.numEpochs - 1]

  epoch = 0
  while epoch < expConfig.numEpochs:
    print "training SP epoch {} ".format(epoch)
    if (expConfig.changeDataSetContinuously or
            epoch == expConfig.changeDataSetAt):
      params['seed'] = epoch
      sdrData.generateInputVectors(params)
      inputVectors = sdrData.getInputVectors()
      numInputVector, inputSize = inputVectors.shape
      testInputs = inputVectors[:numTestInputs, :]

    if expConfig.killInputsAfter > 0 and epoch > expConfig.killInputsAfter:
      if expConfig.topology == 1:
        inputSpaceDim = (params['nX'], params['nY'])
        centerColumn = indexFromCoordinates((15, 15), inputSpaceDim)
        deadInputs = topology.wrappingNeighborhood(centerColumn, 5, inputSpaceDim)
      else:
        zombiePermutation = np.random.permutation(inputSize)
        deadInputs = zombiePermutation[:100]
      inputVectors[:, deadInputs] = 0

    if epoch == expConfig.killCellsAt:
      if expConfig.spatialImp in ['faulty_sp', 'monitored_faulty_sp']:
        if expConfig.topology == 1:
          centerColumn = indexFromCoordinates((15, 15), sp._columnDimensions)
          sp.killCellRegion(centerColumn, 5)
        else:
          sp.killCells(expConfig.killCellPrct)

    if expConfig.trackOverlapCurve:
      noiseLevelList, inputOverlapScore, outputOverlapScore = \
        calculateOverlapCurve(sp, testInputs)
      metrics['noiseRobustness'].append(
        np.trapz(np.flipud(np.mean(outputOverlapScore, 0)),
                 noiseLevelList))
      np.savez(
        './results/input_output_overlap/{}/epoch_{}'.format(expName, epoch),
        noiseLevelList, inputOverlapScore, outputOverlapScore)

    if expConfig.classification:
      # classify SDRs with noise
      noiseLevelList = np.linspace(0, 1.0, 21)
      classification_accuracy = classificationAccuracyVsNoise(
        sp, testInputs, noiseLevelList)
      metrics['classification'].append(
        np.trapz(classification_accuracy, noiseLevelList))
      np.savez('./results/classification/{}/epoch_{}'.format(expName, epoch),
               noiseLevelList, classification_accuracy)

    # train SP here,
    # Learn is turned off at the first epoch to gather stats of untrained SP
    learn = False if epoch == 0 else True

    # randomize the presentation order of input vectors
    sdrOrders = np.random.permutation(np.arange(numInputVector))
    if expConfig.dataSet == 'mnist':
      verbose = True
    else:
      verbose = False

    activeColumnsTrain, meanBoostFactors = runSPOnBatch(sp, inputVectors, learn, sdrOrders, verbose)
    # run SP on test dataset and compute metrics
    activeColumnsPreviousEpoch = copy.copy(activeColumnsCurrentEpoch)
    activeColumnsCurrentEpoch, dum = runSPOnBatch(sp, testInputs, learn=False)

    print "Running discrimination test after training"
    runDiscriminationTest(sp, inputVectors, numPairs=1000)

    stability = calculateStability(activeColumnsCurrentEpoch,
                                   activeColumnsPreviousEpoch)
    if (expConfig.changeDataSetContinuously or
            epoch == expConfig.changeDataSetAt):
      stability = float('nan')
    metrics['stability'].append(stability)

    metrics['sparsity'].append(np.mean(np.mean(activeColumnsCurrentEpoch, 1)))

    metrics['entropy'].append(calculateEntropy(activeColumnsCurrentEpoch))

    # generate ideal SP outputs where all columns have the same activation prob.
    activeColumnsIdeal = np.random.rand(numInputVector, numColumns) > metrics['sparsity'][-1]
    metrics['maxEntropy'].append(calculateEntropy(activeColumnsIdeal))

    connectedSynsPreviousEpoch = copy.copy(connectedSyns)
    sp.getConnectedCounts(connectedCounts)
    connectedSyns = getConnectedSyns(sp)

    metrics['meanBoostFactor'].append(np.mean(meanBoostFactors))
    sp.getActiveDutyCycles(activeDutyCycle)

    metrics['numConnectedSyn'].append(np.sum(connectedCounts))

    numNewSynapses = connectedSyns - connectedSynsPreviousEpoch
    numNewSynapses[numNewSynapses < 0] = 0
    metrics['numNewSyn'].append(np.sum(numNewSynapses))

    numEliminatedSynapses = connectedSynsPreviousEpoch - connectedSyns
    numEliminatedSynapses[numEliminatedSynapses < 0] = 0
    metrics['numRemoveSyn'].append(np.sum(numEliminatedSynapses))

    metrics['reconstructionError'].append(
      reconstructionError(sp, testInputs, activeColumnsCurrentEpoch))

    metrics['witnessError'].append(
      witnessError(sp, testInputs, activeColumnsCurrentEpoch))

    print tabulate(metrics, headers="keys")

    if expConfig.checkRFCenters:
      # check distribution of RF centers, useful to monitor recovery from trauma
      RFcenters, avgDistToCenter = getRFCenters(sp, params, type='connected')
      if expConfig.spatialImp == 'faulty_sp':
        aliveColumns = sp.getAliveColumns()
      else:
        aliveColumns = np.arange(numColumns)
      fig = plotReceptiveFieldCenter(RFcenters[aliveColumns, :],
                                     connectedCounts[aliveColumns],
                                     (params['nX'], params['nY']))
      plt.savefig('figures/RFcenters/{}/epoch_{}.png'.format(expName, epoch))
      plt.close(fig)
      np.savez('results/RFcenters/{}/epoch_{}'.format(expName, epoch),
               RFcenters, avgDistToCenter)

    if expConfig.checkInputSpaceCoverage:
      # check coverage of input space, useful to monitor recovery from trauma
      inputSpaceCoverage = calculateInputSpaceCoverage(sp)
      np.savez('results/InputCoverage/{}/epoch_{}'.format(expName, epoch),
               inputSpaceCoverage, connectedCounts)

      plt.figure(2)
      plt.clf()
      plt.imshow(inputSpaceCoverage, interpolation='nearest', cmap="jet")
      plt.colorbar()
      plt.savefig(
        'figures/InputCoverage/{}/epoch_{}.png'.format(expName, epoch))

    if expConfig.checkTestInput:
      RFcenters, avgDistToCenter = getRFCenters(sp, params, type='connected')
      inputIdx = 0
      outputColumns = np.zeros((numColumns, 1), dtype=uintType)
      sp.compute(testInputs[inputIdx, :], False, outputColumns)
      activeColumns = np.where(outputColumns > 0)[0]
      fig = plotReceptiveFieldCenter(RFcenters[aliveColumns, :],
                                     connectedCounts[aliveColumns],
                                     (params['nX'], params['nY']))
      plt.scatter(RFcenters[activeColumns, 0], RFcenters[activeColumns, 1],
                  color='r')
      plt.savefig(
        'figures/ResponseToTestInputs/{}/epoch_{}.png'.format(expName, epoch))

    if expConfig.saveBoostFactors:
      np.savez('results/boostFactors/{}/epoch_{}'.format(expName, epoch),
               meanBoostFactors)

    if expConfig.showExampleRFs:
      fig = plotReceptiveFields2D(sp, params['nX'], params['nY'])
      plt.savefig('figures/exampleRFs/{}/epoch_{}'.format(expName, epoch))
      plt.close(fig)

    if epoch in checkPoints:
      # inspect SP again
      inspectSpatialPoolerStats(sp, inputVectors, expName+"epoch{}".format(epoch))

    epoch += 1

  # plot stats over training
  fileName = 'figures/network_stats_over_training_{}.pdf'.format(expName)
  plotSPstatsOverTime(metrics, fileName)

  metrics['expName'] = expName
  pickle.dump(metrics, open('results/traces/{}/trace'.format(expName), 'wb'))

  if expConfig.dataSet == 'mnist':
    plotReceptiveFields2D(sp, params['nX'], params['nY'], 41, 6, 6)
  else:
    plotReceptiveFields2D(sp, params['nX'], params['nY'])
  inspectSpatialPoolerStats(sp, inputVectors, inputVectorType + "afterTraining")

  plotExampleInputOutput(sp, inputVectors, expName + "final")
  return metrics, expName



if __name__ == "__main__":
  plt.close('all')

  (expConfig, _args) = _getArgs()

  metrics, expName = runSPexperiments(expConfig)



