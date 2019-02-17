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


"""
Adapted from https://github.com/tugstugi/pytorch-speech-commands
Google speech commands dataset.
"""

import cPickle as pickle
import gc
import itertools
import os

import librosa
import numpy as np
from torch.utils.data import Dataset

__all__ = ['CLASSES', 'SpeechCommandsDataset', 'BackgroundNoiseDataset',
           'PreprocessedSpeechDataset']

# CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
CLASSES = 'unknown, silence, zero, one, two, three, four, five, six, seven, eight, nine'.split(
  ', ')


class SpeechCommandsDataset(Dataset):
  """
  Google speech commands dataset. Only labels in CLASSES, plus silence, are
  treated as known classes. All other classes are used as 'unknown' samples.

  Similar to the Kaggle challenge here:
  https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
  """

  def __init__(self, folder, transform=None, classes=CLASSES,
               silence_percentage=0.1, sample_rate=16000):
    all_classes = [d for d in os.listdir(folder) if
                   os.path.isdir(os.path.join(folder, d)) and not d.startswith(
                     '_')]
    for c in classes[2:]:
      assert c in all_classes

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    for c in all_classes:
      if c not in class_to_idx:
        print ("Class ", c, "assigned as unknown")
        class_to_idx[c] = 0
    data = []
    for c in all_classes:
      d = os.path.join(folder, c)
      target = class_to_idx[c]
      for f in os.listdir(d):
        path = os.path.join(d, f)
        samples, sample_rate = librosa.load(path, sr=sample_rate)
        audio = {'samples': samples, 'sample_rate': sample_rate}
        data.append((audio, target))

    # add silence
    target = class_to_idx['silence']
    samples = np.zeros(sample_rate, dtype=np.float32)
    silence = {'samples': samples, 'sample_rate': sample_rate}
    data += [(silence, target)] * int(len(data) * silence_percentage)

    self.classes = classes
    self.data = data
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    """
    Get item from dataset
    :param index: index in the dataset
    :return: (audio, target) where target is index of the target class.
    :rtype: tuple[dict, int]
    """
    data, target = self.data[index]
    if self.transform is not None:
      data = self.transform(data)

    return data, target

  def make_weights_for_balanced_classes(self):
    """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

    nclasses = len(self.classes)
    count = np.ones(nclasses)
    for item in self.data:
      count[item[1]] += 1

    N = float(sum(count))
    weight_per_class = N / count
    weight = np.zeros(len(self))
    for idx, item in enumerate(self.data):
      weight[idx] = weight_per_class[item[1]]
    return weight


class BackgroundNoiseDataset(Dataset):
  """Dataset for silence / background noise."""

  def __init__(self, folder, transform=None, sample_rate=16000,
               sample_length=1):
    audio_files = [d for d in os.listdir(folder) if
                   os.path.isfile(os.path.join(folder, d)) and d.endswith(
                     '.wav')]
    samples = []
    for f in audio_files:
      path = os.path.join(folder, f)
      s, sr = librosa.load(path, sample_rate)
      samples.append(s)

    samples = np.hstack(samples)
    c = int(sample_rate * sample_length)
    r = len(samples) // c
    self.samples = samples[:r * c].reshape(-1, c)
    self.sample_rate = sample_rate
    self.classes = CLASSES
    self.transform = transform
    self.path = folder

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    data = {'samples': self.samples[index], 'sample_rate': self.sample_rate,
            'path': self.path}

    if self.transform is not None:
      data = self.transform(data)

    return data



class PreprocessedSpeechDataset(Dataset):
  """
  Google Speech Commands dataset preprocessed with with all transforms already
  applied. Use the 'process_dataset.py' script to create preprocessed dataset
  """

  def __init__(self, root, subset, classes=CLASSES, silence_percentage=0.1):
    """
    :param root: Dataset root directory
    :param subset: Which dataset subset to use ("train", "test", "valid", "noise")
    :param classes: List of classes to load. See CLASSES for valid options
    :param silence_percentage: Percentage of the dataset to be filled with silence
    """
    self.classes = classes

    self._root = root
    self._subset = subset
    self._silence_percentage = silence_percentage

    self.data = None

    # Circular list of all epochs in this dataset
    epochs = sorted([int(e) for e in os.listdir(root) if e.isdigit()])
    self._all_epochs = itertools.cycle(epochs)

    # load first epoch
    self.next_epoch()


  def __len__(self):
    return len(self.data)


  def __getitem__(self, index):
    """
    Get item from dataset
    :param index: index in the dataset
    :return: (audio, target) where target is index of the target class.
    :rtype: tuple[dict, int]
    """
    return self.data[index]


  def next_epoch(self):
    """
    Load next epoch from disk
    """
    epoch = next(self._all_epochs)
    folder = os.path.join(self._root, str(epoch), self._subset)
    self.data = []
    silence = None

    gc.disable()

    for filename in os.listdir(folder):
      command = os.path.splitext(os.path.basename(filename))[0]
      with open(os.path.join(folder, filename), "r") as pkl_file:
        audio = pickle.load(pkl_file)

      # Check for 'silence'
      if command == "silence":
        silence = audio
      else:
        target = self.classes.index(os.path.basename(command))
        self.data.extend(itertools.product(audio, [target]))

    gc.enable()

    target = self.classes.index("silence")
    self.data += [(silence, target)] * int(len(self.data) * self._silence_percentage)
    return epoch


  def make_weights_for_balanced_classes(self):
    """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

    nclasses = len(self.classes)
    count = np.ones(nclasses)
    for item in self.data:
      count[item[1]] += 1

    N = float(sum(count))
    weight_per_class = N / count
    weight = np.zeros(len(self.data))
    for idx, item in enumerate(self.data):
      weight[idx] = weight_per_class[item[1]]
    return weight


  @staticmethod
  def isValid(folder, epoch=0):
    """
    Check if the given folder is a valid preprocessed dataset
    """
    # Validate by checking for the training 'silence.pkl' on the given epoch
    # This file is unique to our pre-processed dataset generated by 'process_dataset.py'
    return os.path.exists(os.path.join(folder, str(epoch), "train", "silence.pkl"))
