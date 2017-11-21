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

from htmresearch.support.sp_paper_utils import *

import matplotlib as mpl
from scipy.optimize import curve_fit
mpl.rcParams['pdf.fonttype'] = 42

def nakaRushton(x, c50):
  n=5
  c = 1-x
  y = 1/(1 + 1/(c/c50)**n)
  return y

expName = 'randomSDRVaryingSparsityContinuousLearning_seed_41'
plt.figure()
legendList = []
epochCheck = [0, 5, 10, 20, 40]
for epoch in epochCheck:
  nrData = np.load \
    ('./results/input_output_overlap/{}/epoch_{}.npz'.format(expName, epoch))
  noiseLevelList =  nrData['arr_0']
  inputOverlapScore =  np.mean(nrData['arr_1'], 0)
  outputOverlapScore = np.mean( nrData['arr_2'], 0)
  plt.plot(noiseLevelList, outputOverlapScore)

  popt, pcov = curve_fit(nakaRushton, noiseLevelList, outputOverlapScore,
                         p0=[0.5])
  yfit = nakaRushton(noiseLevelList, popt)
  # plt.plot(noiseLevelList, yfit, 'k--')

  legendList.append('epoch {}'.format(epoch))
plt.legend(legendList)
plt.xlabel('Noise Level')
plt.ylabel('Change of SP output')
plt.savefig('./figures/noise_robustness_{}_beforeChange.pdf'.format(expName))


expName = 'randomSDRVaryingSparsityContinuousLearning_seed_41'
changeDataAt = 50
plt.figure()
legendList = []
epochCheck = [changeDataAt-1, changeDataAt, 119]
for epoch in epochCheck:
  nrData = np.load(
    './results/input_output_overlap/{}/epoch_{}.npz'.format(expName, epoch))
  noiseLevelList = nrData['arr_0']
  inputOverlapScore = nrData['arr_1']
  outputOverlapScore = np.mean(nrData['arr_2'], 0)
  plt.plot(noiseLevelList, outputOverlapScore)
  legendList.append('epoch {}'.format(epoch))
plt.legend(legendList)
plt.xlabel('Noise Level')
plt.ylabel('Change of SP output')
plt.savefig('./figures/noise_robustness_{}.pdf'.format(expName))