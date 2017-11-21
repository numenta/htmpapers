
from matplotlib import pyplot as plt
from htmresearch.support.sequence_learning_utils import *
import pandas as pd

from pylab import rcParams
from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder

rcParams.update({'figure.autolayout': True})
rcParams.update({'figure.facecolor': 'white'})
rcParams.update({'ytick.labelsize': 8})
rcParams.update({'figure.figsize': (12, 6)})
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')

window = 960
skipTrain = 10000
figPath = './result/'

def getDatetimeAxis():
  """
  use datetime as x-axis
  """
  dataSet = 'nyc_taxi'
  filePath = './data/' + dataSet + '.csv'
  data = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                     names=['datetime', 'value', 'timeofday', 'dayofweek'])

  xaxisDate = pd.to_datetime(data['datetime'])
  return xaxisDate


def computeAltMAPE(truth, prediction, startFrom=0):
  return np.nanmean(np.abs(truth[startFrom:] - prediction[startFrom:]))/np.nanmean(np.abs(truth[startFrom:]))


def computeNRMSE(truth, prediction, startFrom=0):
  squareDeviation = computeSquareDeviation(prediction, truth)
  squareDeviation[:startFrom] = None
  return np.sqrt(np.nanmean(squareDeviation))/np.nanstd(truth)


def loadExperimentResult(filePath):
  expResult = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                            names=['step', 'value', 'prediction5'])
  groundTruth = np.roll(expResult['value'], -5)
  prediction5step = np.array(expResult['prediction5'])
  return (groundTruth, prediction5step)



if __name__ == "__main__":
  xaxisDate = getDatetimeAxis()

  ### Figure 2: Continuous LSTM with different window size

  fig = plt.figure()
  dataSet = 'nyc_taxi'
  classifierType = 'SDRClassifierRegion'
  filePath = './prediction/' + dataSet + '_TM_pred.csv'

  trainSPList = [False, True, True]
  boost = [1, 0, 20]
  mapeTMList1 = []
  mapeTMList2 = []
  for i in range(len(boost)):
    tmPrediction = np.load(
      './results/nyc_taxi/{}{}TMprediction_SPLearning_{}_boost_{}.npz'.format(
      dataSet, classifierType, trainSPList[i], boost[i]))

    tmPredictionLL = tmPrediction['arr_0']
    tmPointEstimate = tmPrediction['arr_1']
    tmTruth = tmPrediction['arr_2']

    # encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
    # negLL = computeLikelihood(tmPredictionLL, tmTruth, encoder)
    # negLL[:50] = None
    # negLLTM = plotAccuracy((negLL, xaxisDate), tmTruth,
    #                        window=window, errorType='negLL', label='HTM_trainSP_{}'.format(trainSP))
    #

    absDiff = np.abs(tmTruth - tmPointEstimate)
    mapeTM = plotAccuracy((absDiff, xaxisDate),
                           tmTruth,
                           skipRecordNum = 20,
                           window=window,
                           errorType='mape',
                           label='TrainSP_{}, r={}'.format(trainSPList[i], boost[i]))
    normFactor = np.nanmean(np.abs(tmTruth))
    mapeTMList1.append(np.nanmean(mapeTM[:10000]) / normFactor)
    mapeTMList2.append(np.nanmean(mapeTM[10000:]) / normFactor)
    altMAPETM = computeAltMAPE(tmTruth, tmPointEstimate, 10000)
    print "trainSP {} MAPE {}".format(trainSPList[i], altMAPETM)

  plt.legend()
  plt.savefig('figures/nyc_taxi_performance.pdf')

  rcParams.update({'figure.figsize': (4, 6)})
  plt.figure()
  plt.subplot(221)
  plt.bar(range(3), mapeTMList1)
  plt.ylim([0, .14])
  plt.subplot(222)
  plt.bar(range(3), mapeTMList2)
  plt.ylim([0, .14])
  plt.savefig('figures/nyc_taxi_performance_summary.pdf')
