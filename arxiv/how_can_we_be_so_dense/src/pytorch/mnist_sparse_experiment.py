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

from __future__ import print_function
import os
import random
import sys
import traceback
import numpy as np
import time

import torch
import torch.optim as optim
from torchvision import datasets, transforms

from expsuite import PyExperimentSuite

from pytorch.image_transforms import RandomNoise
from pytorch.sparse_net import SparseNet
from pytorch.duty_cycle_metrics import plotDutyCycles
from pytorch.dataset_utils import createValidationDataSampler
from pytorch.model_utils import evaluateModel, trainModel


class MNISTSparseExperiment(PyExperimentSuite):
  """
  Allows running multiple sparse MNIST experiments in parallel
  """



  def reset(self, params, repetition):
    """
    Called once at the beginning of each experiment.
    """
    self.startTime = time.time()
    print(params)
    seed = params["seed"] + repetition
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Get our directories correct
    self.dataDir = params["datadir"]
    if params.get("create_plots", False):
      self.resultsDir = os.path.join(params["path"], params["name"], "plots")

      if not os.path.exists(self.resultsDir):
        os.makedirs(self.resultsDir)

    self.use_cuda = not params["no_cuda"] and torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(self.dataDir, train=True, download=True,
                                   transform=transform)

    # Create training and validation sampler from MNIST dataset by training on
    # random X% of the training set and validating on the remaining (1-X)%,
    # where X can be tuned via the "validation" parameter
    validation = params.get("validation", 50000.0 / 60000.0)
    if validation < 1.0:
      self.train_sampler, self.validation_sampler = createValidationDataSampler(
        train_dataset, validation)

      self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=params["batch_size"],
                                                      sampler=self.train_sampler)

      self.validation_loader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=params["batch_size"],
                                                           sampler=self.validation_sampler)
    else:
      # No validation. Normal training dataset
      self.validation_loader = None
      self.validation_sampler = None
      self.train_sampler = None
      self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=params["batch_size"],
                                                      shuffle=True)


    self.test_loader = torch.utils.data.DataLoader(
      datasets.MNIST(self.dataDir, train=False, transform=transform),
      batch_size=params["test_batch_size"], shuffle=True)

    # Parse 'n' and 'k' parameters
    n = params["n"]
    k = params["k"]
    if isinstance(n, basestring):
      n = map(int, n.split("_"))
    if isinstance(k, basestring):
      k = map(int, k.split("_"))

    if params["use_cnn"]:
      c1_out_channels = params["c1_out_channels"]
      c1_k = params["c1_k"]
      if isinstance(c1_out_channels, basestring):
        c1_out_channels = map(int, c1_out_channels.split("_"))
      if isinstance(c1_k, basestring):
        c1_k = map(int, c1_k.split("_"))

      # Parse 'c1_input_shape; parameter
      if "c1_input_shape" in params:
        c1_input_shape = map(int, params["c1_input_shape"].split("_"))
      else:
        c1_input_shape = (1, 28, 28)

      sp_model = SparseNet(
        inputSize=c1_input_shape,
        outChannels=c1_out_channels,
        c_k=c1_k,
        kernelSize=5,
        stride=1,
        dropout=params["dropout"],
        n=n,
        k=k,
        outputSize=10,
        boostStrength=params["boost_strength"],
        weightSparsity=params["weight_sparsity"],
        weightSparsityCNN=params["weight_sparsity_cnn"],
        boostStrengthFactor=params["boost_strength_factor"],
        kInferenceFactor=params["k_inference_factor"],
        useBatchNorm=params["use_batch_norm"],
        normalizeWeights=params.get("normalize_weights", False)
      )
    else:
      sp_model = SparseNet(
        n=n,
        k=k,
        outputSize=10,
        boostStrength=params["boost_strength"],
        weightSparsity=params["weight_sparsity"],
        weightSparsityCNN=params["weight_sparsity_cnn"],
        boostStrengthFactor=params["boost_strength_factor"],
        kInferenceFactor=params["k_inference_factor"],
        dropout=params["dropout"],
        useBatchNorm=params["use_batch_norm"],
        normalizeWeights=params.get("normalize_weights", False)
      )
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs")
      sp_model = torch.nn.DataParallel(sp_model)

    self.model = sp_model.to(self.device)
    self.learningRate = params["learning_rate"]
    self.optimizer = self.createOptimizer(params, self.model)
    self.lr_scheduler = self.createLearningRateScheduler(params, self.optimizer)


  def iterate(self, params, repetition, iteration):
    """
    Called once for each training iteration (== epoch here).
    """
    try:
      print("\nStarting iteration",iteration)
      t1 = time.time()
      ret = {}

      # Update learning rate using learning rate scheduler if configured
      if self.lr_scheduler is not None:
        # ReduceLROnPlateau lr_scheduler step should be called after validation,
        # all other lr_schedulers should be called before training
        if params["lr_scheduler"] != "ReduceLROnPlateau":
          self.lr_scheduler.step()

      self.train(params, epoch=iteration)

      # Run validation test
      if self.validation_loader is not None:
        validation = self.test(params, self.validation_loader)

        # ReduceLROnPlateau step should be called after validation
        if params["lr_scheduler"] == "ReduceLROnPlateau":
          self.lr_scheduler.step(validation["test_loss"])

        ret["validation"] = validation
        print("Validation: Test error=", validation["testerror"],
              "entropy=", validation["entropy"])

      # Run noise test
      if (params["test_noise_every_epoch"] or
          iteration == params["iterations"] - 1):
        ret.update(self.runNoiseTests(params))
        print("Noise test results: totalCorrect=", ret["totalCorrect"],
              "Test error=", ret["testerror"], ", entropy=", ret["entropy"])
        if ret["totalCorrect"] > 100000 and ret["testerror"] > 98.3:
          print("*******")
          print(params)

      ret.update({"elapsedTime": time.time() - self.startTime})
      ret.update({"learningRate": self.learningRate if self.lr_scheduler is None
                                                    else self.lr_scheduler.get_lr()})

      print("Iteration time= {0:.3f} secs, "
            "total elapsed time= {1:.3f} mins".format(
              time.time() - t1,ret["elapsedTime"]/60.0))

    except Exception as e:
      # Tracebacks are not printed if using multiprocessing so we do it here
      tb = sys.exc_info()[2]
      traceback.print_tb(tb)
      raise RuntimeError("Something went wrong in iterate", e)

    return ret


  def finalize(self, params, rep):
    """
    Called once we are done.
    """
    if params.get("savenet", True):
      # Save the full model once we are done.
      saveDir = os.path.join(params["path"], params["name"], "model.pt")
      torch.save(self.model, saveDir)


  def createLearningRateScheduler(self, params, optimizer):
    """
    Creates the learning rate scheduler and attach the optimizer
    """
    lr_scheduler = params.get("lr_scheduler", None)
    if lr_scheduler is None:
      return None

    if lr_scheduler == "StepLR":
      lr_scheduler_params = "{'step_size': 1, 'gamma':" + str(params["learning_rate_factor"]) + "}"

    else:
      lr_scheduler_params = params.get("lr_scheduler_params", None)
      if lr_scheduler_params is None:
        raise ValueError("Missing 'lr_scheduler_params' for {}".format(lr_scheduler))

    # Get lr_scheduler class by name
    clazz = eval("torch.optim.lr_scheduler.{}".format(lr_scheduler))

    # Parse scheduler parameters from config
    lr_scheduler_params = eval(lr_scheduler_params)

    return clazz(optimizer, **lr_scheduler_params)

  def createOptimizer(self, params, model):
    """
    Create a new instance of the optimizer
    """
    lr = params["learning_rate"]
    print("Creating optimizer with learning rate=", lr)
    if params["optimizer"] == "SGD":
      optimizer = optim.SGD(model.parameters(), lr=lr,
                            momentum=params["momentum"])
    elif params["optimizer"] == "Adam":
      optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
      raise LookupError("Incorrect optimizer value")

    return optimizer

  def train(self, params, epoch):
    """
    Train one epoch of this model by iterating through mini batches. An epoch
    ends after one pass through the training set, or if the number of mini
    batches exceeds the parameter "batches_in_epoch".
    """

    # Callback used to log information on every batch
    def log(model, batch_idx):
      if batch_idx % params["log_interval"] == 0:
        entropy = model.entropy()
        print("logging: {} learning iterations, entropy: {} / {}".format(
          model.getLearningIterations(), float(entropy), model.maxEntropy()))

        if params["create_plots"]:
          plotDutyCycles(model.dutyCycle,
                         self.resultsDir + "/figure_" + str(epoch) + "_" +
                         str(model.getLearningIterations()))


    # Adjust first epoch batch size to stabilize the dutycycles at the
    # beginning of the training
    loader = self.train_loader
    batches_in_epoch = params["batches_in_epoch"]
    if "first_epoch_batch_size" in params:
      if epoch == 0:
        batches_in_epoch = params.get("batches_in_first_epoch", batches_in_epoch)
        loader = torch.utils.data.DataLoader(self.train_loader.dataset,
                                             batch_size=params["first_epoch_batch_size"],
                                             sampler=self.train_loader.sampler)

    trainModel(model=self.model, loader=loader,
               optimizer=self.optimizer, device=self.device,
               batches_in_epoch=batches_in_epoch,
               batch_callback=log)

    self.model.postEpoch()


  def test(self, params, test_loader):
    """
    Test the model using the given loader and return test metrics
    """
    results = evaluateModel(model=self.model, device=self.device,
                            loader=test_loader)
    entropy = self.model.entropy()
    ret = {"num_correct": results["total_correct"],
           "test_loss": results["loss"],
           "testerror": results["accuracy"] * 100,
           "entropy": float(entropy)}


    return ret


  def runNoiseTests(self, params):
    """
    Test the model with different noise values and return test metrics.
    """
    ret = {}

    # Noise on validation data
    validation = {} if self.validation_sampler is not None else None

    # Test with noise
    total_correct = 0
    validation_total_correct = 0
    for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
      transform = transforms.Compose([
        transforms.ToTensor(),
        RandomNoise(noise, whiteValue=0.1307 + 2*0.3081),
        transforms.Normalize((0.1307,), (0.3081,))
      ])
      test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(self.dataDir, train=False, transform=transform),
        batch_size=params["test_batch_size"], shuffle=True)

      testResult = self.test(params, test_loader)
      total_correct += testResult["num_correct"]
      ret[noise]= testResult

      if validation is not None:
        validation_loader = torch.utils.data.DataLoader(
          datasets.MNIST(self.dataDir, train=True, transform=transform),
          sampler=self.validation_sampler,
          batch_size=params["test_batch_size"])

        validationResult = self.test(params, validation_loader)
        validation_total_correct += validationResult["num_correct"]
        validation[noise] = validationResult

    ret["totalCorrect"] = total_correct
    ret["testerror"] = ret[0.0]["testerror"]
    ret["entropy"] = ret[0.0]["entropy"]

    if "nonzeros" in ret[0.0]:
      ret["nonzeros"] = ret[0.0]["nonzeros"]

    if validation is not None:
      validation["totalCorrect"] = validation_total_correct
      validation["testerror"] = validation[0.0]["testerror"]
      validation["entropy"] = validation[0.0]["entropy"]
      ret["validation"] = validation

    return ret



if __name__ == '__main__':
  suite = MNISTSparseExperiment()
  suite.start()
