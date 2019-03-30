# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from __future__ import print_function

import os

import numpy as np
from tabulate import tabulate

from expsuite import PyExperimentSuite

def bestScore(scores):
  """
  Given a single repetition of a single experiment, return the test, and
  total noise score from the epoch with maximum test accuracy.
  """
  testScores, totalCorrect = scores[0], scores[1]

  bestEpoch = np.argmax(testScores)
  maxTestScore = testScores[bestEpoch]
  maxNoiseScore = totalCorrect[bestEpoch]

  return maxTestScore, bestEpoch, maxNoiseScore


def getErrorBars(expPath, suite):
  """
  Go through each epoch in each repetition in this path. For each repetition
  select the epoch with the best test score as the best epoch. Collect the test
  score and noise score for that epoch, as the optimal for that repetition.

  Return the overall mean, and stdev for test accuracy and noise accuracy across
  the optimal values for each repetition.
  """

  # Get the iteration with maximum validation accuracy.
  results = suite.get_all_histories_over_repetitions(
    exp=expPath,
    tags=["testerror", "totalCorrect"])

  numExps = len(results["testerror"])

  testScores = np.zeros(numExps)
  noiseScores = np.zeros(numExps)
  for i,scoresForRepetition in enumerate(
          zip(results["testerror"], results["totalCorrect"])):
    maxTestScore, bestEpoch, maxNoiseScore = bestScore(scoresForRepetition)
    testScores[i] = maxTestScore
    noiseScores[i] = maxNoiseScore

  return {
    "test_score": (testScores.mean(), testScores.std()),
    "noise_score": (noiseScores.mean(), noiseScores.std())
  }



if __name__ == '__main__':
  suite = PyExperimentSuite()
  suite.parse_opt()
  suite.parse_cfg()
  experiments = suite.options.experiments or suite.cfgparser.sections()

  testScoresTable = [["Network", "Test Score", "Noise Score"]]
  for name in experiments:
    exps = suite.get_exps(suite.get_exp(name)[0])
    for exp in exps:
      if not os.path.exists(exp):
        continue

      errorBars = getErrorBars(exp, suite)
      test_score = u"{0:.2f} ± {1:.2f}".format(*errorBars["test_score"])
      noise_score = u"{0:,.0f} ± {1:.2f}".format(*errorBars["noise_score"])

      params = suite.get_params(exp=exp)
      testScoresTable.append([params["name"], test_score, noise_score])

  print()
  print(tabulate(testScoresTable, headers="firstrow", tablefmt="grid"))
  print()
  print(tabulate(testScoresTable, headers="firstrow", tablefmt="latex"))
