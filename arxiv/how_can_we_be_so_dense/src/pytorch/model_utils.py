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
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import logging
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR)



def trainModel(model, loader, optimizer, device, criterion=F.nll_loss,
               batches_in_epoch=sys.maxsize, batch_callback=None,
               progress_bar=None):
  """
  Train the given model by iterating through mini batches. An epoch
  ends after one pass through the training set, or if the number of mini
  batches exceeds the parameter "batches_in_epoch".

  :param model: pytorch model to be trained
  :type model: torch.nn.Module
  :param loader: train dataset loader
  :type loader: :class:`torch.utils.data.DataLoader`
  :param optimizer: Optimizer object used to train the model.
         This function will train the model on every batch using this optimizer
         and the :func:`torch.nn.functional.nll_loss` function
  :param batches_in_epoch: Max number of mini batches to train.
  :param device: device to use ('cpu' or 'cuda')
  :type device: :class:`torch.device
  :param criterion: loss function to use
  :type criterion: function
  :param batch_callback: Callback function to be called on every batch with the
                         following parameters: model, batch_idx
  :type batch_callback: function
  :param progress_bar: Optional :class:`tqdm` progress bar args.
                       None for no progress bar
  :type progress_bar: dict or None
  """
  model.train()
  if progress_bar is not None:
    loader = tqdm(loader, **progress_bar)
    # update progress bar total based on batches_in_epoch
    if batches_in_epoch < len(loader):
      loader.total = batches_in_epoch

  for batch_idx, (data, target) in enumerate(loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if batch_callback is not None:
      batch_callback(model=model, batch_idx=batch_idx)
    if batch_idx >= batches_in_epoch:
      break

  if progress_bar is not None:
    loader.n = loader.total
    loader.close()



def evaluateModel(model, loader, device, criterion=F.nll_loss, progress=None):
  """
  Evaluate pre-trained model using given test dataset loader.

  :param model: Pretrained pytorch model
  :type model: torch.nn.Module
  :param loader: test dataset loader
  :type loader: :class:`torch.utils.data.DataLoader`
  :param device: device to use ('cpu' or 'cuda')
  :type device: :class:`torch.device
  :param criterion: loss function to use
  :type criterion: function
  :param progress: Optional :class:`tqdm` progress bar args. None for no progress bar
  :type progress: dict or None

  :return: dictionary with computed "accuracy", "loss", "total_correct". The
           loss value is computed using :func:`torch.nn.functional.nll_loss`
  :rtype: dict
  """
  model.eval()
  loss = 0
  correct = 0
  dataset_len = len(loader.sampler)

  if progress is not None:
    loader = tqdm(loader, **progress)

  with torch.no_grad():
    for data, target in loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      loss += criterion(output, target, reduction='sum').item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  if progress is not None:
    loader.close()

  loss /= dataset_len
  accuracy = correct / dataset_len

  return {"total_correct": correct,
          "loss": loss,
          "accuracy": accuracy}
