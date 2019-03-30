# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

from __future__ import print_function, division

import collections
import torch

import torch.nn as nn

import pytorch.modules as htm



class SparseNet(nn.Module):

  def __init__(self,
               n=2000,
               k=200,
               outChannels=0,
               c_k=0,
               kernelSize=5,
               stride=1,
               inputSize=28*28,
               outputSize=10,
               kInferenceFactor=1.0,
               weightSparsity=0.5,
               weightSparsityCNN=0.5,
               boostStrength=1.0,
               boostStrengthFactor=1.0,
               dropout=0.0,
               useBatchNorm=True,
               normalizeWeights=False,
               useSoftmax=True,
               padding=0,
               maxPoolKernel=2):
    """
    A network with one or more hidden layers, which can be a sequence of
    k-sparse CNN followed by a sequence of k-sparse linear layer with optional
    dropout layers in between the k-sparse linear layers.

        [CNNSDR] x len(outChannels)
            |
        [Flatten]
            |
        [LinearSDR => Dropout] x len(n)
            |
        [Linear => Softmax]

    :param n:
      Number of units in each fully connected k-sparse linear layer.
      Use 0 to disable the linear layer
    :type n: int or list[int]

    :param k:
      Number of ON (non-zero) units per iteration in each k-sparse linear layer.
      The sparsity of this layer will be k / n. If k >= n, the layer acts as a
      traditional fully connected RELU layer
    :type k: int or list[int]

    :param outChannels:
      Number of channels (filters) in each k-sparse convolutional layer.
      Use 0 to disable the CNN layer
    :type outChannels: int or list[int]

    :param c_k:
      Number of ON (non-zero) units per iteration in each k-sparse convolutional
      layer. The sparsity of this layer will be c_k / c_n. If c_k >= c_n, the
      layer acts as a traditional convolutional layer.
    :type c_k: int or list[int]

    :param kernelSize:
      Kernel size to use in each k-sparse convolutional layer.
    :type kernelSize: int or list[int]

    :param stride:
      Stride value to use in each k-sparse convolutional layer.
    :type stride: int or list[int]

    :param inputSize:
      If the CNN layer is enable this parameter holds a tuple representing
      (in_channels,height,width). Otherwise it will hold the total
      dimensionality of input vector of the first linear layer. We apply
      view(-1, inputSize) to the data before passing it to Linear layers.
    :type inputSize: int or tuple[int,int,int]

    :param outputSize:
      Total dimensionality of output vector
    :type outputSize: int

    :param kInferenceFactor:
      During inference (training=False) we increase k by this factor.
    :type kInferenceFactor: float

    :param weightSparsity:
      Pct of weights that are allowed to be non-zero in each linear layer.
    :type weightSparsity: float or list[float]

    :param weightSparsityCNN:
      Pct of weights that are allowed to be non-zero in each convolutional layer.
    :type weightSparsityCNN: float or list[float]

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      boost strength is multiplied by this factor after each epoch.
      A value < 1.0 will decrement it every epoch.
    :type boostStrengthFactor: float

    :param dropout:
      dropout probability used to train the second and subsequent layers.
      A value 0.0 implies no dropout
    :type dropout: float

    :param useBatchNorm:
      If True, applies batchNorm for each layer.
    :type useBatchNorm: bool

    :param normalizeWeights:
      If True, each LinearSDR layer will have its weights normalized to the
      number of non-zeros instead of the whole input size
    :type normalizeWeights: bool

    :param useSoftmax:
      If True, use soft max to compute probabilities
    :type useSoftmax: bool

    :param padding:
        cnn layer Zero-padding added to both sides of the input
    :type padding: int

    :param maxPoolKernel:
      The size of the window to take a max over
    :type maxPoolKernel: int
    """
    super(SparseNet, self).__init__()


    # Validate CNN sdr params
    if isinstance(inputSize, collections.Sequence):
      assert inputSize[1] == inputSize[2], "sparseCNN only supports square images"

    if type(outChannels) is not list:
      outChannels = [outChannels]
    if type(c_k) is not list:
      c_k = [c_k] * len(outChannels)
    assert(len(outChannels) == len(c_k))
    if type(kernelSize) is not list:
      kernelSize = [kernelSize] * len(outChannels)
    assert(len(outChannels) == len(kernelSize))
    if type(stride) is not list:
      stride = [stride] * len(outChannels)
    assert(len(outChannels) == len(stride))
    if type(padding) is not list:
      padding = [padding] * len(outChannels)
    assert(len(outChannels) == len(padding))
    if type(weightSparsityCNN) is not list:
      weightSparsityCNN = [weightSparsityCNN] * len(outChannels)
    assert(len(outChannels) == len(weightSparsityCNN))
    for i in range(len(outChannels)):
      assert (weightSparsityCNN[i] >= 0)

    # Validate linear sdr params
    if type(n) is not list:
      n = [n]
    if type(k) is not list:
      k = [k] * len(n)
    assert(len(n) == len(k))
    for i in range(len(n)):
      assert(k[i] <= n[i])
    if type(weightSparsity) is not list:
      weightSparsity = [weightSparsity] * len(n)
    assert(len(n) == len(weightSparsity))
    for i in range(len(n)):
      assert (weightSparsity[i] >= 0)

    self.k = k
    self.kInferenceFactor = kInferenceFactor
    self.n = n
    self.outChannels = outChannels
    self.c_k = c_k
    self.inputSize = inputSize
    self.weightSparsity = weightSparsity   # Pct of weights that are non-zero
    self.boostStrengthFactor = boostStrengthFactor
    self.boostStrength = boostStrength
    self.kernelSize = kernelSize
    self.stride = stride
    self.padding = padding
    self.learningIterations = 0


    inputFeatures = inputSize
    outputLength = inputFeatures
    cnnSdr = nn.Sequential()
    # CNN Layers
    for i in range(len(outChannels)):
      if outChannels[i] != 0:
        inChannels, h, w = inputFeatures
        cnn = nn.Conv2d(in_channels=inChannels,
                        out_channels=outChannels[i],
                        kernel_size=kernelSize[i],
                        padding=padding[i],
                        stride=stride[i])

        if 0 < weightSparsityCNN[i] < 1:
          sparseCNN = htm.SparseWeights2d(cnn, weightSparsityCNN[i])
          cnnSdr.add_module("cnnSdr{}_cnn".format(i + 1), sparseCNN)
        else:
          cnnSdr.add_module("cnnSdr{}_cnn".format(i + 1), cnn)

        # Batch Norm
        if useBatchNorm:
          bn = nn.BatchNorm2d(outChannels[i], affine=False)
          cnnSdr.add_module("cnnSdr{}_bn".format(i + 1), bn)

        # Max pool
        maxpool = nn.MaxPool2d(kernel_size=maxPoolKernel)
        cnnSdr.add_module("cnnSdr{}_maxpool".format(i + 1), maxpool)

        wout = (w + 2 * padding[i] - kernelSize[i]) // stride[i] + 1
        maxpoolWidth = wout // 2
        outputLength = maxpoolWidth * maxpoolWidth * outChannels[i]
        if 0 < c_k[i] < outputLength:
          kwinner = htm.KWinners2d(n=outputLength, k=c_k[i],
                                   channels=outChannels[i],
                                   kInferenceFactor=kInferenceFactor,
                                   boostStrength=boostStrength,
                                   boostStrengthFactor=boostStrengthFactor)
          cnnSdr.add_module("cnnSdr{}_kwinner".format(i + 1), kwinner)
        else:
          cnnSdr.add_module("cnnSdr{}_relu".format(i + 1), nn.ReLU())

        # Feed this layer output into next layer input
        inputFeatures = (outChannels[i], maxpoolWidth, maxpoolWidth)

    if len(cnnSdr) > 0:
      inputFeatures = outputLength
      self.cnnSdr = cnnSdr
    else:
      self.cnnSdr = None

    # Flatten input before passing to linear layers
    self.flatten = htm.Flatten()

    # Linear layers
    self.linearSdr = nn.Sequential()

    for i in range(len(n)):
      if n[i] != 0:
        linear = nn.Linear(inputFeatures, n[i])
        if 0 < weightSparsity[i] < 1:
          linear = htm.SparseWeights(linear, weightSparsity=weightSparsity[i])
          if normalizeWeights:
            linear.apply(htm.normalizeSparseWeights)
        self.linearSdr.add_module("linearSdr{}".format(i + 1), linear)

        if useBatchNorm:
          self.linearSdr.add_module("linearSdr{}_bn".format(i + 1),
                                    nn.BatchNorm1d(n[i], affine=False))

        if dropout > 0.0:
          self.linearSdr.add_module("linearSdr{}_dropout".format(i + 1),
                                    nn.Dropout(dropout))

        if 0 < k[i] < n[i]:
          kwinner = htm.KWinners(n=n[i], k=k[i],
                                 kInferenceFactor=kInferenceFactor,
                                 boostStrength=boostStrength,
                                 boostStrengthFactor=boostStrengthFactor)
          self.linearSdr.add_module("linearSdr{}_kwinner".format(i + 1), kwinner)
        else:
          self.linearSdr.add_module("linearSdr{}_relu".format(i + 1), nn.ReLU())

        # Feed this layer output into next layer input
        inputFeatures = n[i]

    # Add one fully connected layer after all hidden layers
    self.fc = nn.Linear(inputFeatures, outputSize)

    # Use softmax to compute probabilities
    if useSoftmax:
      self.softmax = nn.LogSoftmax(dim=1)
    else:
      self.softmax = None


  def postEpoch(self):
    self.apply(htm.updateBoostStrength)
    self.apply(htm.rezeroWeights)


  def forward(self, x):
    if self.cnnSdr is not None:
      x = self.cnnSdr(x)
    x = self.flatten(x)
    x = self.linearSdr(x)
    x = self.fc(x)

    if self.softmax is not None:
      x = self.softmax(x)

    if self.training:
      batchSize = x.shape[0]
      self.learningIterations += batchSize

    return x


  def getLearningIterations(self):
    return self.learningIterations

  def maxEntropy(self):
    entropy = 0
    for module in self.modules():
      if module == self:
        continue
      if hasattr(module, "maxEntropy"):
        entropy += module.maxEntropy()

    return entropy

  def entropy(self):
    """
    Returns the current entropy
    """
    entropy = 0
    for module in self.modules():
      if module == self:
        continue
      if hasattr(module, "entropy"):
        entropy += module.entropy()

    return entropy


  def pruneWeights(self, minWeight):
    """
    Prune all the weights whose absolute magnitude is less than minWeight
    :param minWeight: min weight to prune. If zero then no pruning
    :type minWeight: float
    """
    if minWeight == 0.0:
      return

    # Collect all weights
    weights = [v for k, v in self.named_parameters() if 'weight' in k]
    for w in weights:
      # Filter weights above threshold
      mask = torch.ge(torch.abs(w.data), minWeight)
      # Zero other weights
      w.data.mul_(mask.type(torch.float32))

  def pruneDutycycles(self, threshold=0.0):
    """
    Prune all the units with dutycycles whose absolute magnitude is less than
    the given threshold
    :param threshold: min threshold to prune. If less than zero then no pruning
    :type threshold: float
    """
    if threshold < 0.0:
      return

    # Collect all layers with 'dutyCycle'
    for m in self.modules():
      if m == self:
        continue
      if hasattr(m, 'pruneDutycycles'):
        m.pruneDutycycles(threshold)
