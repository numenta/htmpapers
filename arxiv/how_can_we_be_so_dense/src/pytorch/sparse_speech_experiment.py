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

import copy
import json
import os
import time

import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


from pytorch.benchmark_utils import (
  register_nonzero_counter, unregister_counter_nonzero)
from expsuite import PyExperimentSuite

from pytorch.sparse_net import SparseNet
from pytorch.duty_cycle_metrics import plotDutyCycles
from pytorch.speech_commands_dataset import (
  SpeechCommandsDataset, BackgroundNoiseDataset, PreprocessedSpeechDataset
)
from pytorch.audio_transforms import *
from pytorch.resnet_models import resnet9

class SparseSpeechExperiment(PyExperimentSuite):
  """
  This experiment tests the Google Speech Commands dataset, available here:
  http://download.tensorflow.org/data/speech_commands_v0.01.tar

  Allows running multiple sparse speech experiments in parallel
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
    self.dataDir = os.path.join(params["datadir"], "speech_commands")
    self.resultsDir = os.path.join(params["path"], params["name"], "plots")

    if not os.path.exists(self.resultsDir):
      os.makedirs(self.resultsDir)

    self.use_cuda = not params["no_cuda"] and torch.cuda.is_available()
    if self.use_cuda:
      print("*********using cuda!")
    self.device = torch.device("cuda" if self.use_cuda else "cpu")

    self.use_preprocessed_dataset = False
    self.loadDatasets(params)

    # Parse 'n' and 'k' parameters
    n = params["n"]
    k = params["k"]
    if isinstance(n, basestring):
      n = map(int, n.split("_"))
    if isinstance(k, basestring):
      k = map(int, k.split("_"))

    if params["model_type"] == "cnn":
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
        c1_input_shape = (1, 32, 32)

      sp_model = SparseNet(
        inputSize=c1_input_shape,
        outputSize=len(self.train_loader.dataset.classes),
        outChannels=c1_out_channels,
        c_k=c1_k,
        kernelSize=5,
        stride=1,
        dropout=params["dropout"],
        n=n,
        k=k,
        boostStrength=params["boost_strength"],
        weightSparsity=params["weight_sparsity"],
        weightSparsityCNN=params["weight_sparsity_cnn"],
        boostStrengthFactor=params["boost_strength_factor"],
        kInferenceFactor=params["k_inference_factor"],
        useBatchNorm=params["use_batch_norm"],
        normalizeWeights=params.get("normalize_weights", False)
      )
    elif params["model_type"] == "resnet9":
      sp_model = resnet9(num_classes=len(self.train_loader.dataset.classes),
                         in_channels=1)
    elif params["model_type"] == "linear":
      sp_model = SparseNet(
        n=n,
        k=k,
        inputSize=32*32,
        outputSize=len(self.train_loader.dataset.classes),
        boostStrength=params["boost_strength"],
        weightSparsity=params["weight_sparsity"],
        boostStrengthFactor=params["boost_strength_factor"],
        kInferenceFactor=params["k_inference_factor"],
        dropout=params["dropout"],
        useBatchNorm=params["use_batch_norm"],
        normalizeWeights=params.get("normalize_weights", False)
      )
    else:
      raise RuntimeError("Unknown model type")

    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs")
      sp_model = torch.nn.DataParallel(sp_model)

    self.model = sp_model.to(self.device)
    self.learningRate = params["learning_rate"]
    self.optimizer = self.createOptimizer(params, self.model)
    self.lr_scheduler = self.createLearningRateScheduler(params, self.optimizer)

    self.best_score = 0.0
    self.best_model = None
    self.best_epoch = -1

  def iterate(self, params, repetition, iteration):
    """
    Called once for each training iteration (== epoch here).
    """
    print("\nStarting iteration",iteration)
    print("Learning rate:", self.learningRate if self.lr_scheduler is None
                                              else self.lr_scheduler.get_lr())
    t1 = time.time()
    ret = {}

    # Update dataset epoch when using pre-processed speech dataset
    if self.use_preprocessed_dataset:
      t2 = time.time()
      self.train_loader.dataset.next_epoch()
      self.validation_loader.dataset.next_epoch()
      self.test_loader.dataset.next_epoch()
      self.bg_noise_loader.dataset.next_epoch()
      print("Dataset Load time = {0:.3f} secs, ".format(time.time() - t2))

    # Update learning rate using learning rate scheduler if configured
    if self.lr_scheduler is not None:
      # ReduceLROnPlateau lr_scheduler step should be called after validation,
      # all other lr_schedulers should be called before training
      if params["lr_scheduler"] != "ReduceLROnPlateau":
        self.lr_scheduler.step()

    self.train(params, epoch=iteration, repetition=repetition)

    # Run validation test
    if self.validation_loader is not None:
      validation = self.test(params, self.validation_loader)

      # ReduceLROnPlateau step should be called after validation
      if params["lr_scheduler"] == "ReduceLROnPlateau":
        self.lr_scheduler.step(validation["test_loss"])

      ret["validation"] = validation
      print("Validation: error=", validation["testerror"],
            "entropy=", validation["entropy"],
            "loss=", validation["test_loss"])
      ret.update({"validationerror": validation["testerror"]})

    # Run test set
    if self.test_loader is not None:
      testResults = self.test(params, self.test_loader)
      ret["testResults"] = testResults
      print("Test: error=", testResults["testerror"],
            "entropy=", testResults["entropy"],
            "loss=", testResults["test_loss"])
      ret.update({"testerror": testResults["testerror"]})

      score = testResults["testerror"]
      if score > self.best_score:
        self.best_epoch = iteration
        self.best_score = score
        self.best_model = copy.deepcopy(self.model)


    # Run bg noise set
    if self.bg_noise_loader is not None:
      bgResults = self.test(params, self.bg_noise_loader)
      ret["bgResults"] = bgResults
      print("BG noise error=", bgResults["testerror"])
      ret.update({"bgerror": bgResults["testerror"]})

    ret.update({"elapsedTime": time.time() - self.startTime})
    ret.update({"learningRate": self.learningRate if self.lr_scheduler is None
                                                  else self.lr_scheduler.get_lr()})

    # Run noise set
    if params.get("run_noise_tests", False):
      ret.update(self.runNoiseTests(params))
      print("Noise test results: totalCorrect=", ret["totalCorrect"],
            "Test error=", ret["testerror"], ", entropy=", ret["entropy"])

    ret.update({"elapsedTime": time.time() - self.startTime})
    ret.update({"learningRate": self.learningRate if self.lr_scheduler is None
                                                  else self.lr_scheduler.get_lr()})

    print("Iteration time= {0:.3f} secs, "
          "total elapsed time= {1:.3f} mins".format(
            time.time() - t1,ret["elapsedTime"]/60.0))

    return ret


  def finalize(self, params, rep):
    """
    Save the full model once we are done.
    """
    if params.get("saveNet", True):
      saveDir = os.path.join(params["path"], params["name"],
                             "best_model_{}.pt".format(rep))
      torch.save(self.best_model, saveDir)

    # Run noise test on best model at the end
    if params.get("run_noise_tests_best_model", False):
      self.model = self.best_model
      results = self.runNoiseTests(params)

      # Update bets epoch log with noise results
      fullpath = os.path.join(params['path'], params['name'])
      logfile = os.path.join(fullpath, '%i.log' % rep)
      log = []

      with open(logfile, 'r') as f:
        for line in f:
          log.append(json.loads(line))

      with open(logfile, 'w') as f:
        log[self.best_epoch].update(results)
        for entry in log:
          json.dump(entry, f)
          f.write('\n')
          f.flush()



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
                            momentum=params["momentum"],
                            weight_decay=params["weight_decay"],
                            )
    elif params["optimizer"] == "Adam":
      optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
      raise LookupError("Incorrect optimizer value")

    return optimizer


  def train(self, params, epoch, repetition):
    """
    Train one epoch of this model by iterating through mini batches. An epoch
    ends after one pass through the training set, or if the number of mini
    batches exceeds the parameter "batches_in_epoch".
    """
    # Check for pre-trained model
    modelCheckpoint = os.path.join(params["path"], params["name"],
                                   "model_{}_{}.pt".format(repetition, epoch))
    if os.path.exists(modelCheckpoint):
      self.model = torch.load(modelCheckpoint, map_location=self.device)
      return

    self.model.train()
    for batch_idx, (batch, target) in enumerate(self.train_loader):
      data = batch["input"]
      if params["model_type"] in ["resnet9", "cnn"]:
        data = torch.unsqueeze(data, 1)
      data, target = data.to(self.device), target.to(self.device)
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      self.optimizer.step()

      # Log info every log_interval mini batches
      if batch_idx % params["log_interval"] == 0:
        entropy = self.model.entropy()
        print(
          "logging: ",self.model.getLearningIterations(),
          " learning iterations, elapsedTime", time.time() - self.startTime,
          " entropy:", float(entropy)," / ", self.model.maxEntropy(),
          "loss:", loss.item())
        if params["create_plots"]:
          plotDutyCycles(self.model.dutyCycle,
                         self.resultsDir + "/figure_"+str(epoch)+"_"+str(
                           self.model.getLearningIterations()))

      if batch_idx >= params["batches_in_epoch"]:
        break

    self.model.postEpoch()

    # Save model checkpoint on every epoch
    if params.get("save_every_epoch", False):
      torch.save(self.model, modelCheckpoint)



  def test(self, params, test_loader):
    """
    Test the model using the given loader and return test metrics
    """
    self.model.eval()
    test_loss = 0
    correct = 0

    nonzeros = None
    count_nonzeros = params.get("count_nonzeros", False)
    if count_nonzeros:
      nonzeros = {}
      register_nonzero_counter(self.model, nonzeros)

    with torch.no_grad():
      for batch, target in test_loader:
        data = batch["input"]
        if params["model_type"] in ["resnet9", "cnn"]:
          data = torch.unsqueeze(data, 1)
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        # count nonzeros only once
        if count_nonzeros:
          count_nonzeros = False
          unregister_counter_nonzero(self.model)

    test_loss /= len(test_loader.sampler)
    test_error = 100. * correct / len(test_loader.sampler)

    entropy = self.model.entropy()
    ret = {"num_correct": correct,
           "test_loss": test_loss,
           "testerror": test_error,
           "entropy": float(entropy)}

    if nonzeros is not None:
      ret["nonzeros"] = nonzeros

    return ret


  def runNoiseTests(self, params):
    """
    Test the model with different noise values and return test metrics.
    """
    if self.use_preprocessed_dataset:
      raise NotImplementedError("Noise tests requires raw data")

    ret = {}
    testDataDir = os.path.join(self.dataDir, "test")
    n_mels = 32

    # Test with noise
    total_correct = 0

    for noise in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:

      # Create noise dataset with noise transform
      noiseTransform = transforms.Compose([
        FixAudioLength(),
        AddNoise(noise),
        ToSTFT(),
        ToMelSpectrogramFromSTFT(n_mels=n_mels),
        DeleteSTFT(),
        ToTensor('mel_spectrogram', 'input')
      ])

      noiseDataset = SpeechCommandsDataset(
        testDataDir,
        noiseTransform,
        silence_percentage=0,
      )

      noise_loader = DataLoader(noiseDataset,
                                 batch_size=params["batch_size"],
                                 sampler=None,
                                 shuffle=False,
                                 pin_memory=self.use_cuda,
                                 )

      testResult = self.test(params, noise_loader)
      total_correct += testResult["num_correct"]
      ret[noise]= testResult

    ret["totalCorrect"] = total_correct
    ret["testerror"] = ret[0.0]["testerror"]
    ret["entropy"] = ret[0.0]["entropy"]

    if "nonzeros" in ret[0.0]:
      ret["nonzeros"] = ret[0.0]["nonzeros"]

    return ret


  def loadDatasets(self, params):
    """
    The GSC dataset specifies specific files to be used as training, test,
    and validation.  We assume the data has already been processed according
    to those files into separate train, test, and valid directories.

    For our experiment we use a subset of the data (10 categories out of 30),
    just like the Kaggle competition.
    """
    n_mels = 32

    # Check if using pre-processed data or raw data
    self.use_preprocessed_dataset = PreprocessedSpeechDataset.isValid(self.dataDir)
    if self.use_preprocessed_dataset:
      trainDataset = PreprocessedSpeechDataset(self.dataDir, subset="train")
      validationDataset = PreprocessedSpeechDataset(self.dataDir, subset="valid",
                                                    silence_percentage=0)
      testDataset = PreprocessedSpeechDataset(self.dataDir, subset="test",
                                              silence_percentage=0)
      bgNoiseDataset = PreprocessedSpeechDataset(self.dataDir, subset="noise",
                                                 silence_percentage=0)
    else:
      trainDataDir = os.path.join(self.dataDir, "train")
      testDataDir = os.path.join(self.dataDir, "test")
      validationDataDir = os.path.join(self.dataDir, "valid")
      backgroundNoiseDir = os.path.join(self.dataDir, params["background_noise_dir"])

      dataAugmentationTransform = transforms.Compose([
        ChangeAmplitude(),
        ChangeSpeedAndPitchAudio(),
        FixAudioLength(),
        ToSTFT(),
        StretchAudioOnSTFT(),
        TimeshiftAudioOnSTFT(),
        FixSTFTDimension(),
      ])

      featureTransform = transforms.Compose(
        [
          ToMelSpectrogramFromSTFT(n_mels=n_mels),
          DeleteSTFT(),
          ToTensor('mel_spectrogram', 'input')
        ])

      trainDataset = SpeechCommandsDataset(
        trainDataDir,
        transforms.Compose([
          dataAugmentationTransform,
          # add_bg_noise,               # Uncomment to allow adding BG noise
                                        # during training
          featureTransform
        ]))

      testFeatureTransform = transforms.Compose([
        FixAudioLength(),
        ToMelSpectrogram(n_mels=n_mels),
        ToTensor('mel_spectrogram', 'input')
      ])

      validationDataset = SpeechCommandsDataset(
        validationDataDir,
        testFeatureTransform,
        silence_percentage=0,
      )

      testDataset = SpeechCommandsDataset(
        testDataDir,
        testFeatureTransform,
        silence_percentage=0,
      )

      bg_dataset = BackgroundNoiseDataset(
        backgroundNoiseDir,
        transforms.Compose([FixAudioLength(), ToSTFT()]),
      )

      bgNoiseTransform = transforms.Compose([
        FixAudioLength(),
        ToSTFT(),
        AddBackgroundNoiseOnSTFT(bg_dataset),
        ToMelSpectrogramFromSTFT(n_mels=n_mels),
        DeleteSTFT(),
        ToTensor('mel_spectrogram', 'input')
      ])

      bgNoiseDataset = SpeechCommandsDataset(
        testDataDir,
        bgNoiseTransform,
        silence_percentage=0,
      )

    weights = trainDataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))

    # print("Number of training samples=",len(trainDataset))
    # print("Number of validation samples=",len(validationDataset))
    # print("Number of test samples=",len(testDataset))

    self.train_loader = DataLoader(trainDataset,
                                   batch_size=params["batch_size"],
                                   sampler=sampler
                                   )

    self.validation_loader = DataLoader(validationDataset,
                                        batch_size=params["batch_size"],
                                        shuffle=False
                                        )

    self.test_loader = DataLoader(testDataset,
                                  batch_size=params["batch_size"],
                                  sampler=None,
                                  shuffle=False
                                  )

    self.bg_noise_loader = DataLoader(bgNoiseDataset,
                                  batch_size=params["batch_size"],
                                  sampler=None,
                                  shuffle=False
                                  )



if __name__ == '__main__':
  suite = SparseSpeechExperiment()
  suite.start()
