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
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.duty_cycle_metrics import (
  maxEntropy, binaryEntropy
)
from pytorch.functions import k_winners, k_winners2d



def updateBoostStrength(m):
  """
  Function used to update KWinner modules boost strength after each epoch.

  Call using :meth:`torch.nn.Module.apply` after each epoch if required
  For example: ``m.apply(updateBoostStrength)``

  :param m: KWinner module
  """
  if isinstance(m, KWinnersBase):
    if m.training:
      m.boostStrength = m.boostStrength * m.boostStrengthFactor



class KWinnersBase(nn.Module):
  """
  Base KWinners class
  """
  __metaclass__ = abc.ABCMeta


  def __init__(self, n, k, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """
    :param n:
      Number of units
    :type n: int

    :param k:
      The activity of the top k units will be allowed to remain, the rest are set
      to zero
    :type k: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """
    super(KWinnersBase, self).__init__()
    assert (boostStrength >= 0.0)

    self.n = n
    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.learningIterations = 0

    # Boosting related parameters
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor
    self.dutyCyclePeriod = dutyCyclePeriod


  def getLearningIterations(self):
    return self.learningIterations


  @abc.abstractmethod
  def updateDutyCycle(self, x):
    """
     Updates our duty cycle estimates with the new value. Duty cycles are
     updated according to the following formula:

    .. math::
        dutyCycle = \\frac{dutyCycle \\times \\left( period - batchSize \\right)
                            + newValue}{period}
    :param x:
      Current activity of each unit
    """
    raise NotImplementedError


  def updateBoostStrength(self):
    """
    Update boost strength using given strength factor during training
    """
    if self.training:
      self.boostStrength = self.boostStrength * self.boostStrengthFactor


  def entropy(self):
    """
    Returns the current total entropy of this layer
    """
    if self.k < self.n:
      _, entropy = binaryEntropy(self.dutyCycle)
      return entropy
    else:
      return 0


  def maxEntropy(self):
    """
    Returns the maximum total entropy we can expect from this layer
    """
    return maxEntropy(self.n, self.k)



class KWinners(KWinnersBase):
  """
  Applies K-Winner function to the input tensor

  See :class:`htmresearch.frameworks.pytorch.functions.k_winners`

  """


  def __init__(self, n, k, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """
    :param n:
      Number of units
    :type n: int

    :param k:
      The activity of the top k units will be allowed to remain, the rest are set
      to zero
    :type k: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """

    super(KWinners, self).__init__(n=n, k=k,
                                   kInferenceFactor=kInferenceFactor,
                                   boostStrength=boostStrength,
                                   boostStrengthFactor=boostStrengthFactor,
                                   dutyCyclePeriod=dutyCyclePeriod)
    self.register_buffer("dutyCycle", torch.zeros(self.n))


  def forward(self, x):
    # Apply k-winner algorithm if k < n, otherwise default to standard RELU
    if self.k >= self.n:
      return F.relu(x)

    if self.training:
      k = self.k
    else:
      k = min(int(round(self.k * self.kInferenceFactor)), self.n)

    x = k_winners.apply(x, self.dutyCycle, k, self.boostStrength)

    if self.training:
      self.updateDutyCycle(x)

    return x


  def updateDutyCycle(self, x):
    batchSize = x.shape[0]
    self.learningIterations += batchSize
    period = min(self.dutyCyclePeriod, self.learningIterations)
    self.dutyCycle.mul_(period - batchSize)
    self.dutyCycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
    self.dutyCycle.div_(period)



class KWinners2d(KWinnersBase):
  """
  Applies K-Winner function to the input tensor

  See :class:`htmresearch.frameworks.pytorch.functions.k_winners2d`

  """


  def __init__(self, n, k, channels, kInferenceFactor=1.0, boostStrength=1.0,
               boostStrengthFactor=1.0, dutyCyclePeriod=1000):
    """

    :param n:
      Number of units. Usually the output of the max pool or whichever layer
      preceding the KWinners2d layer.
    :type n: int

    :param k:
      The activity of the top k units will be allowed to remain, the rest are set
      to zero
    :type k: int

    :param channels:
      Number of channels (filters) in the convolutional layer.
    :type channels: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int
    """
    super(KWinners2d, self).__init__(n=n, k=k,
                                     kInferenceFactor=kInferenceFactor,
                                     boostStrength=boostStrength,
                                     boostStrengthFactor=boostStrengthFactor,
                                     dutyCyclePeriod=dutyCyclePeriod)

    self.channels = channels
    self.register_buffer("dutyCycle", torch.zeros((1, channels, 1, 1)))


  def forward(self, x):
    # Apply k-winner algorithm if k < n, otherwise default to standard RELU
    if self.k >= self.n:
      return F.relu(x)

    if self.training:
      k = self.k
    else:
      k = min(int(round(self.k * self.kInferenceFactor)), self.n)

    x = k_winners2d.apply(x, self.dutyCycle, k, self.boostStrength)

    if self.training:
      self.updateDutyCycle(x)

    return x


  def updateDutyCycle(self, x):
    batchSize = x.shape[0]
    self.learningIterations += batchSize

    scaleFactor = float(x.shape[2] * x.shape[3])
    period = min(self.dutyCyclePeriod, self.learningIterations)
    self.dutyCycle.mul_(period - batchSize)
    s = x.gt(0).sum(dim=(0, 2, 3), dtype=torch.float) / scaleFactor
    self.dutyCycle.reshape(-1).add_(s)
    self.dutyCycle.div_(period)


  def entropy(self):
    entropy = super(KWinners2d, self).entropy()
    return entropy * self.n / self.channels
