## ----------------------------------------------------------------------
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


import importlib
import os
from optparse import OptionParser
import yaml

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.frameworks.opf.prediction_metrics_manager import MetricsManager

from nupic.frameworks.opf import metrics
from nupic.frameworks.opf.htm_prediction_model import HTMPredictionModel

import pandas as pd
from htmresearch.support.sequence_learning_utils import *
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['pdf.fonttype'] = 42

plt.ion()

DATA_DIR = "../../htmresearch/data"
MODEL_PARAMS_DIR = "./model_params"

def getMetricSpecs(predictedField, stepsAhead=5):
  _METRIC_SPECS = (
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'negativeLogLikelihood',
                         'window': 1000, 'steps': stepsAhead}),
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'nrmse', 'window': 1000,
                         'steps': stepsAhead}),
  )
  return _METRIC_SPECS


def createModel(modelParams):
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": predictedField})
  return model


def getModelParamsFromName(dataSet):
  # importName = "model_params.%s_model_params" % (
  #   dataSet.replace(" ", "_").replace("-", "_")
  # )
  # print "Importing model params from %s" % importName
  try:
    importedModelParams = yaml.safe_load(
      open('model_params/nyc_taxi_model_params.yaml'))
    # importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. Run swarm first!"
                    % dataSet)
  return importedModelParams


def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='nyc_taxi',
                    dest="dataSet",
                    help="DataSet Name, choose from rec-center-hourly, nyc_taxi")

  parser.add_option("-p",
                    "--plot",
                    default=False,
                    dest="plot",
                    help="Set to True to plot result")

  parser.add_option("--stepsAhead",
                    help="How many steps ahead to predict. [default: %default]",
                    default=5,
                    type=int)

  parser.add_option("--trainSP",
                    help="Whether to train SP",
                    default=True,
                    dest="trainSP",
                    type=int)

  parser.add_option("--boostStrength",
                    help="strength of boosting",
                    default=1,
                    dest="boostStrength",
                    type=int)

  parser.add_option("-c",
                    "--classifier",
                    type=str,
                    default='SDRClassifierRegion',
                    dest="classifier",
                    help="Classifier Type: SDRClassifierRegion or CLAClassifierRegion")

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder


def getInputRecord(df, predictedField, i):
  inputRecord = {
    predictedField: float(df[predictedField][i]),
    "timeofday": float(df["timeofday"][i]),
    "dayofweek": float(df["dayofweek"][i]),
  }
  return inputRecord


def printTPRegionParams(tpregion):
  """
  Note: assumes we are using TemporalMemory/TPShim in the TPRegion
  """
  tm = tpregion.getSelf()._tfdr
  print "------------PY  TemporalMemory Parameters ------------------"
  print "numberOfCols             =", tm.getColumnDimensions()
  print "cellsPerColumn           =", tm.getCellsPerColumn()
  print "minThreshold             =", tm.getMinThreshold()
  print "activationThreshold      =", tm.getActivationThreshold()
  print "newSynapseCount          =", tm.getMaxNewSynapseCount()
  print "initialPerm              =", tm.getInitialPermanence()
  print "connectedPerm            =", tm.getConnectedPermanence()
  print "permanenceInc            =", tm.getPermanenceIncrement()
  print "permanenceDec            =", tm.getPermanenceDecrement()
  print "predictedSegmentDecrement=", tm.getPredictedSegmentDecrement()
  print



def runMultiplePass(df, model, nMultiplePass, nTrain):
  """
  run CLA model through data record 0:nTrain nMultiplePass passes
  """
  predictedField = model.getInferenceArgs()['predictedField']
  print "run TM through the train data multiple times"
  for nPass in xrange(nMultiplePass):
    for j in xrange(nTrain):
      inputRecord = getInputRecord(df, predictedField, j)
      result = model.run(inputRecord)
      if j % 100 == 0:
        print " pass %i, record %i" % (nPass, j)
    # reset temporal memory
    model._getTPRegion().getSelf()._tfdr.reset()

  return model



def runMultiplePassSPonly(df, model, nMultiplePass, nTrain):
  """
  run CLA model SP through data record 0:nTrain nMultiplePass passes
  """

  predictedField = model.getInferenceArgs()['predictedField']
  print "run TM through the train data multiple times"
  for nPass in xrange(nMultiplePass):
    for j in xrange(nTrain):
      inputRecord = getInputRecord(df, predictedField, j)
      model._sensorCompute(inputRecord)
      model._spCompute()
      if j % 400 == 0:
        print " pass %i, record %i" % (nPass, j)

  return model


if __name__ == "__main__":
  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  plot = _options.plot
  classifierType = _options.classifier
  trainSP = bool(_options.trainSP)
  boostStrength = _options.boostStrength

  DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
  predictedField = "passenger_count"

  modelParams = getModelParamsFromName("nyc_taxi")

  modelParams['modelParams']['clParams']['steps'] = str(_options.stepsAhead)
  modelParams['modelParams']['clParams']['regionName'] = classifierType
  modelParams['modelParams']['spParams']['boostStrength'] = boostStrength

  print "Creating model from %s..." % dataSet

  # use customized CLA model
  model = HTMPredictionModel(**modelParams['modelParams'])
  model.enableInference({"predictedField": predictedField})
  model.enableLearning()
  model._spLearningEnabled = bool(trainSP)
  model._tpLearningEnabled = True

  print model._spLearningEnabled
  printTPRegionParams(model._getTPRegion())

  inputData = "%s/%s.csv" % (DATA_DIR, dataSet.replace(" ", "_"))

  sensor = model._getSensorRegion()
  encoderList = sensor.getSelf().encoder.getEncoderList()
  if sensor.getSelf().disabledEncoder is not None:
    classifier_encoder = sensor.getSelf().disabledEncoder.getEncoderList()
    classifier_encoder = classifier_encoder[0]
  else:
    classifier_encoder = None

  _METRIC_SPECS = getMetricSpecs(predictedField, stepsAhead=_options.stepsAhead)
  metric = metrics.getModule(_METRIC_SPECS[0])
  metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                  model.getInferenceType())

  if plot:
    plotCount = 1
    plotHeight = max(plotCount * 3, 6)
    fig = plt.figure(figsize=(14, plotHeight))
    gs = gridspec.GridSpec(plotCount, 1)
    plt.title(predictedField)
    plt.ylabel('Data')
    plt.xlabel('Timed')
    plt.tight_layout()
    plt.ion()

  print "Load dataset: ", dataSet
  df = pd.read_csv(inputData, header=0, skiprows=[1, 2])

  nTrain = 5000

  maxBucket = classifier_encoder.n - classifier_encoder.w + 1
  likelihoodsVecAll = np.zeros((maxBucket, len(df)))

  prediction_nstep = None
  time_step = []
  actual_data = []
  patternNZ_track = []
  predict_data = np.zeros((_options.stepsAhead, 0))
  predict_data_ML = []
  negLL_track = []

  activeCellNum = []
  trueBucketIndex = []
  sp = model._getSPRegion().getSelf()._sfdr
  spActiveCellsCount = np.zeros(sp.getColumnDimensions())

  for i in xrange(len(df)):
    inputRecord = getInputRecord(df, predictedField, i)
    result = model.run(inputRecord)
    trueBucketIndex.append(model._getClassifierInputRecord(inputRecord).bucketIndex)

    # inspect SP
    sp = model._getSPRegion().getSelf()._sfdr
    spOutput = model._getSPRegion().getOutputData('bottomUpOut')
    spActiveCellsCount[spOutput.nonzero()[0]] += 1

    tp = model._getTPRegion()
    tm = tp.getSelf()._tfdr
    activeColumn = tm.getActiveCells()
    activeCellNum.append(len(activeColumn))

    result.metrics = metricsManager.update(result)

    negLL = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
               "field=%s"%(_options.stepsAhead, predictedField)]
    if i % 100 == 0 and i>0:
      negLL = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='negativeLogLikelihood':steps=%d:window=1000:"
               "field=%s"%(_options.stepsAhead, predictedField)]
      nrmse = result.metrics["multiStepBestPredictions:multiStep:"
               "errorMetric='nrmse':steps=%d:window=1000:"
               "field=%s"%(_options.stepsAhead, predictedField)]

      numActiveCell = np.mean(activeCellNum[-100:])

      print "After %i records, %d-step negLL=%f nrmse=%f ActiveCell %f " % \
            (i, _options.stepsAhead, negLL, nrmse, numActiveCell)

    last_prediction = prediction_nstep
    prediction_nstep = \
      result.inferences["multiStepBestPredictions"][_options.stepsAhead]

    bucketLL = \
      result.inferences['multiStepBucketLikelihoods'][_options.stepsAhead]
    likelihoodsVec = np.zeros((maxBucket,))
    if bucketLL is not None:
      for (k, v) in bucketLL.items():
        likelihoodsVec[k] = v

    time_step.append(i)
    actual_data.append(inputRecord[predictedField])
    predict_data_ML.append(
      result.inferences['multiStepBestPredictions'][_options.stepsAhead])
    negLL_track.append(negLL)

    likelihoodsVecAll[0:len(likelihoodsVec), i] = likelihoodsVec


  predData_TM_n_step = np.roll(np.array(predict_data_ML), _options.stepsAhead)
  nTest = len(actual_data) - nTrain - _options.stepsAhead
  NRMSE_TM = NRMSE(actual_data[nTrain:nTrain+nTest], predData_TM_n_step[nTrain:nTrain+nTest])
  print "NRMSE on test data: ", NRMSE_TM


  # calculate neg-likelihood
  predictions = np.transpose(likelihoodsVecAll)
  truth = np.roll(actual_data, -5)

  from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
  encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)

  bucketIndex2 = []
  negLL  = []
  minProb = 0.0001
  for i in xrange(len(truth)):
    bucketIndex2.append(np.where(encoder.encode(truth[i]))[0])
    outOfBucketProb = 1 - sum(predictions[i,:])
    prob = predictions[i, bucketIndex2[i]]
    if prob == 0:
      prob = outOfBucketProb
    if prob < minProb:
      prob = minProb
    negLL.append( -np.log(prob))

  negLL = computeLikelihood(predictions, truth, encoder)
  negLL[:5000] = np.nan
  x = range(len(negLL))

  if not os.path.exists("./results/nyc_taxi/"):
    os.makedirs("./results/nyc_taxi/")
  np.savez('./results/nyc_taxi/{}{}TMprediction_SPLearning_{}_boost_{}'.format(
    dataSet, classifierType, trainSP, boostStrength),
    predictions, predict_data_ML, truth)

  activeDutyCycle = np.zeros(sp.getColumnDimensions(), dtype=np.float32)
  sp.getActiveDutyCycles(activeDutyCycle)
  overlapDutyCycle = np.zeros(sp.getColumnDimensions(), dtype=np.float32)
  sp.getOverlapDutyCycles(overlapDutyCycle)

  if not os.path.exists("./figures/nyc_taxi/"):
    os.makedirs("./figures/nyc_taxi/")
  plt.figure()
  plt.clf()
  plt.subplot(2, 2, 1)
  plt.hist(overlapDutyCycle)
  plt.xlabel('overlapDutyCycle')

  plt.subplot(2, 2, 2)
  plt.hist(activeDutyCycle)
  plt.xlim([0, .1])
  plt.xlabel('activeDutyCycle-1000')

  plt.subplot(2, 2, 3)
  totalActiveDutyCycle = spActiveCellsCount.astype('float32') / len(df)
  dutyCycleDist, binEdge = np.histogram(totalActiveDutyCycle,
                                        bins=20, range=[-0.0025, 0.0975])
  dutyCycleDist = dutyCycleDist.astype('float32')/np.sum(dutyCycleDist)
  binWidth = np.mean(binEdge[1:]-binEdge[:-1])
  binCenter = binEdge[:-1] + binWidth/2
  plt.bar(binCenter, dutyCycleDist, width=0.005)
  plt.xlim([-0.0025, .1])
  plt.ylim([0, .7])
  plt.xlabel('activeDutyCycle-Total')
  plt.savefig('figures/nyc_taxi/DutyCycle_SPLearning_{}_boost_{}.pdf'.format(
    trainSP, boostStrength))


