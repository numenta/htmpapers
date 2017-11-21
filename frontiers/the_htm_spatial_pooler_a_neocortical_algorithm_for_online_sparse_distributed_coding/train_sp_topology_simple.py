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
A simple use case of SP with topology for profiling purpose
"""
import copy
from optparse import OptionParser
import pprint


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from nupic.algorithms.spatial_pooler import SpatialPooler as PYSpatialPooler

from htmresearch.support.spatial_pooler_monitor_mixin import (
  SpatialPoolerMonitorMixin)

class MonitoredSpatialPooler(SpatialPoolerMonitorMixin,
                             PYSpatialPooler): pass

from nupic.bindings.algorithms import SpatialPooler as CPPSpatialPooler
from nupic.bindings.math import GetNTAReal
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
  else:
    raise RuntimeError("Invalide spatialImp")
  return sp



def getSpatialPoolerParams(params, boosting=False):
  if boosting is False:
    from sp_params import spParamTopologyWithBoosting as spatialPoolerParameters
  else:
    from sp_params import spParamTopologyNoBoosting as spatialPoolerParameters

  spatialPoolerParameters['inputDimensions'] = (params['nX'], params['nY'])

  print "Spatial Pooler Parameters: "
  pprint.pprint(spatialPoolerParameters)
  return spatialPoolerParameters


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
                    default=1,
                    dest="boosting",
                    help="Whether to use boosting")

  parser.add_option("-e",
                    "--numEpochs",
                    type=int,
                    default=100,
                    dest="numEpochs",
                    help="number of epochs")

  parser.add_option("--spatialImp",
                    type=str,
                    default="cpp",
                    dest="spatialImp",
                    help="spatial pooler implementations: py, c++, or "
                         "monitored_sp")

  (options, remainder) = parser.parse_args()
  print options
  return options, remainder



if __name__ == "__main__":
  (_options, _args) = _getArgs()
  inputVectorType = _options.dataSet
  numEpochs = _options.numEpochs
  spatialImp = _options.spatialImp

  inputVectorType = 'randomBarSets'
  params = {'dataType': 'randomBarSets',
            'numInputVectors': 100,
            'nX': 40,
            'nY': 40,
            'barHalfLength': 3,
            'numBarsPerInput': 4,
            'seed': 41}

  sdrData = SDRDataSet(params)
  inputVectors = sdrData.getInputVectors()
  numInputVector, inputSize = inputVectors.shape

  print "Training Data Size {} Dimensions {}".format(numInputVector, inputSize)

  spParams = getSpatialPoolerParams(params, boosting=False)

  sp = createSpatialPooler(spatialImp, spParams)
  columnNumber = np.prod(sp.getColumnDimensions())

  for epoch in range(numEpochs):
    print "training SP epoch {}".format(epoch)
    # train SP here,
    # randomize the presentation order of input vectors
    sdrOrders = np.random.permutation(np.arange(numInputVector))
    for i in range(numInputVector):
      outputColumns = np.zeros((columnNumber, 1), dtype=uintType)
      inputVector = copy.deepcopy(inputVectors[sdrOrders[i]][:])

      sp.compute(inputVector, True, outputColumns)
