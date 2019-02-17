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
"""
Pre-process google commands dataset applying multiple transformations
to the original .wav files and saving them as pickled dictionaries
"""

from __future__ import print_function

import argparse
import cPickle as pickle
import gc
import itertools
import multiprocessing as mp
import os
import sys
import traceback

from torchvision import transforms

from pytorch.audio_transforms import *
from pytorch.speech_commands_dataset import BackgroundNoiseDataset

# Multiprocess shared variable used to update the progress
progress = None



def transform_folder(args):
  """
  Transform all the files in the source dataset for the given command and save
  the results as a single pickle file in the destination dataset
  :param args: tuple with the following arguments:
               - the command name: 'zero', 'one', 'two', ...
               - transforms to apply to wav file
               - full path of the source dataset
               - full path of the destination dataset
  """
  command, (transform, src, dest) = args
  try:
    print(progress.value, "remaining")

    # Apply transformations to all files
    data = []
    data_dir = os.path.join(src, command)
    for filename in os.listdir(data_dir):
      path = os.path.join(data_dir, filename)
      data.append(transform({'path': path}))

    # Save results
    pickleFile = os.path.join(dest, "{}.pkl".format(command))
    gc.disable()
    with open(pickleFile, "wb") as f:
      pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    gc.enable()

    # Update progress
    with progress.get_lock():
      progress.value -= 1

  except Exception as e:
    print(command, e, file=sys.stderr)
    traceback.print_exc()



def main():
  parser = argparse.ArgumentParser(
    description='Pre-process google commands dataset.')
  parser.add_argument('--source', '-s', type=str, required=True,
                      help='the path to the root folder of the google commands '
                           'train dataset.')
  parser.add_argument('--dest', '-d', type=str, required=True,
                      help='the path where to stored the transformed dataset')
  parser.add_argument('-sample_rate', '-sr', type=int,
                      default=16000,
                      help='target sampling rate')
  args = parser.parse_args()

  # Dataset folders
  noise_folder = os.path.join(args.source, '_background_noise_')
  train_folder = os.path.join(args.source, 'train')
  valid_folder = os.path.join(args.source, 'valid')
  test_folder = os.path.join(args.source, 'test')

  dest_noise_folder = os.path.join(args.dest, 'noise')
  dest_train_folder = os.path.join(args.dest, 'train')
  dest_valid_folder = os.path.join(args.dest, 'valid')
  dest_test_folder = os.path.join(args.dest, 'test')

  # Dataset transforms
  n_mels = 32
  trainingTransform = transforms.Compose([
    LoadAudio(),
    ChangeAmplitude(),
    ChangeSpeedAndPitchAudio(),
    FixAudioLength(),
    ToSTFT(),
    StretchAudioOnSTFT(),
    TimeshiftAudioOnSTFT(),
    FixSTFTDimension(),
    ToMelSpectrogramFromSTFT(n_mels=n_mels),
    DeleteSTFT(),
    ToTensor('mel_spectrogram', 'input')
  ])
  testFeatureTransform = transforms.Compose([
    LoadAudio(),
    FixAudioLength(),
    ToMelSpectrogram(n_mels=n_mels),
    ToTensor('mel_spectrogram', 'input')
  ])

  bg_dataset = BackgroundNoiseDataset(
    noise_folder,
    transforms.Compose([FixAudioLength(), ToSTFT()]),
  )
  bgNoiseTransform = transforms.Compose([
    LoadAudio(),
    FixAudioLength(),
    ToSTFT(),
    AddBackgroundNoiseOnSTFT(bg_dataset),
    ToMelSpectrogramFromSTFT(n_mels=n_mels),
    DeleteSTFT(),
    ToTensor('mel_spectrogram', 'input')
  ])


  # Create transformation tuples in the following format: (transform, src, dest)
  transformations = zip(
    [trainingTransform, testFeatureTransform, testFeatureTransform, bgNoiseTransform],
    [train_folder, test_folder, valid_folder, test_folder],
    [dest_train_folder, dest_test_folder, dest_valid_folder, dest_noise_folder])

  # Prepare transformations parameters for background process pool applying
  # transformations on every command
  params = list(itertools.product(['zero', 'one', 'two', 'three', 'four',
                                   'five', 'six', 'seven', 'eight', 'nine'],
                                  transformations))

  # Create destination folders before starting the background processes to avoid
  # race condition
  destination_folders = [dest_train_folder, dest_test_folder, dest_valid_folder,
                         dest_noise_folder]
  for folder in destination_folders:
    if not os.path.exists(folder):
      os.makedirs(folder)

  # Run transformations in parallel updating the shared 'progress' variable
  global progress
  progress = mp.Value('i', len(params))
  pool = mp.Pool()
  pool.map(transform_folder, params)

  # Save transformed silence
  with open("{}/silence.pkl".format(dest_train_folder), "wb") as f:
    silence = trainingTransform({'path': '', 'sample_rate': args.sample_rate})
    pickle.dump(silence, f, pickle.HIGHEST_PROTOCOL)

  with open("{}/silence.pkl".format(dest_noise_folder), "wb") as f:
    silence = bgNoiseTransform({'path': '', 'sample_rate': args.sample_rate})
    pickle.dump(silence, f, pickle.HIGHEST_PROTOCOL)

  silence = testFeatureTransform({'path': '', 'sample_rate': args.sample_rate})
  with open("{}/silence.pkl".format(dest_valid_folder), "wb") as f:
    pickle.dump(silence, f, pickle.HIGHEST_PROTOCOL)

  with open("{}/silence.pkl".format(dest_test_folder), "wb") as f:
    pickle.dump(silence, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
  main()
