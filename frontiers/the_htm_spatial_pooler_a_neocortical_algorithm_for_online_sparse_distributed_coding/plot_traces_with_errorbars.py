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

"""
Plot continuous learning experiment result with error bars
Run './runRepeatedExperiment.sh' in terminal before using this script
"""

import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import rcParams
from scipy import stats


mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')


def convertToNumpyArray(trace):
  for k in trace.keys():
    if k == 'expName':
      continue
    n = len(trace[k])
    trace[k] = np.reshape(np.array(trace[k]), (n, 1))
  return trace



def concatenateTraces(trace1, trace2):
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

  for k in metrics.keys():
    metrics[k] = np.concatenate((np.array(trace1[k]),
                                 np.array(trace2[k])), 1)
  return metrics



def calculateMeanStd(trace):
  meanTrace = np.mean(trace, axis=1)
  stdTrace = np.std(trace, axis=1)
  return (meanTrace, stdTrace)



def plotBarWithErr(ax, y, yerr, ylabel, xtickLabels):
  inds = np.arange(len(y))
  ax.bar(inds+.2, y, yerr=yerr, width=0.6)
  ax.set_ylabel(ylabel)
  ax.set_xticks(inds+.5)
  ax.set_xticklabels(xtickLabels)
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_rotation('vertical')



if __name__ == "__main__":

  traceAll = None
  changeDataAt = 50

  for seed in range(1, 11):
    expName = 'randomSDRVaryingSparsityContinuousLearning_seed_{}'.format(seed)
    trace = pickle.load(open('./results/traces/{}/trace'.format(expName), 'rb'))
    trace = convertToNumpyArray(trace)

    if traceAll is None:
      traceAll = trace
    else:
      traceAll = concatenateTraces(traceAll, trace)

  traceAll['stability'][changeDataAt, :] = traceAll['stability'][changeDataAt-1, :]
  tracesToPlot = ['stability', 'entropy', 'noiseRobustness',
                  'numNewSyn', 'numRemoveSyn']
  ylabelList = ['Stability', 'Entropy (bits)', 'Noise Robustness',
                'Synapses Formation', 'Synapse Removal']
  numEpochs, numRpts = traceAll['entropy'].shape

  fig, axs = plt.subplots(nrows=len(tracesToPlot), ncols=1, sharex=True)
  for i in range(len(tracesToPlot)):
    traceName = tracesToPlot[i]
    (mean, std) = calculateMeanStd(traceAll[traceName])
    color = 'k'

    x = range(numEpochs)
    axs[i].fill_between(x, mean - std, mean + std,
                    alpha=0.3, edgecolor=color, facecolor=color)
    axs[i].plot(x, mean, color, color=color, linewidth=.5)
    axs[i].set_ylabel(ylabelList[i])

    axs[i].plot([changeDataAt, changeDataAt], axs[i].get_ylim(), 'k--')

  # adjust axis limit and tick spacings
  for i in [3, 4]:
    yl = axs[i].get_ylim()
    axs[i].set_ylim([0, yl[1]])
  axs[0].set_yticks(np.linspace(.6, 1, 5))
  axs[1].set_yticks(np.linspace(.08, .14, 4))

  axs[4].set_xlabel('Epochs')
  plt.savefig('figures/ContinuousLearning_WithErrBars.pdf')

  rcParams.update({'figure.autolayout': True})
  fig, ax = plt.subplots(nrows=1, ncols=3)
  checkPoints = [0, 49, 50, 119]
  meanEntropy = np.mean(traceAll['entropy'][checkPoints, :], 1)
  stdEntropy = np.std(traceAll['entropy'][checkPoints, :], 1)
  maxEntropy = np.mean(np.mean(traceAll['maxEntropy'][10:, :], 1))

  print "test entropy difference before/after learning: "
  print stats.ttest_rel(traceAll['entropy'][0, :], traceAll['entropy'][49, :])

  meanNoiseRobustness = np.mean(traceAll['noiseRobustness'][checkPoints, :], 1)
  stdNoiseRobustness = np.std(traceAll['noiseRobustness'][checkPoints, :], 1)
  xtickLabels = ['Before Training', 'Before Change', 'After Change',
                 'After Recovery']

  print "test noise robustness before/after learning: "
  print stats.ttest_rel(traceAll['noiseRobustness'][0, :], traceAll['noiseRobustness'][49, :])

  plotBarWithErr(ax[0], meanEntropy, stdEntropy, 'Entropy (bits)', xtickLabels)
  ax[0].plot(ax[0].get_xlim(), [maxEntropy, maxEntropy], 'k--')

  plotBarWithErr(ax[1], meanNoiseRobustness, stdNoiseRobustness,
                 'Noise Robustness', xtickLabels)
  plt.savefig('figures/ContinuousLearning_BarChart.pdf')
