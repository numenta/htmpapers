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
Splits the google speech commands into train, validation and test sets.

In the small case, it will only create files for the categories listed
in smallCategories.  In this case it will create a subdirectory for all files
that are unused.

"""

from __future__ import print_function

import os
import shutil
import argparse


smallCategories = ['eight', 'nine', 'three', 'one', 'zero',
                   'seven', 'two', 'six', 'five', 'four']


def remove_unused_directories(src_folder, categories):
  """
  list all directories.
  Remove all that are not in categories list (excluding background noise)
  :param src_folder:
  :param categories:
  :return:
  """
  files = os.listdir(src_folder)
  os.mkdir("unused")
  for name in files:
    full_path = os.path.join(src_folder, name)
    if os.path.isdir(full_path):
      if not( name in categories or name == "_background_noise_"):
        newPath = os.path.join("unused", name)
        print("moving: ", full_path, "-->", newPath)
        shutil.move(full_path, newPath)


def move_files(src_folder, to_folder, list_file):
  with open(list_file) as f:
    for line in f.readlines():
      line = line.rstrip()
      dirname = os.path.dirname(line)
      dest = os.path.join(to_folder, dirname)
      if not os.path.exists(dest):
        os.mkdir(dest)
      shutil.move(os.path.join(src_folder, line), dest)


def move_directories(src_folder, to_folder, categories):
  for name in categories:
    src = os.path.join(src_folder, name)
    newPath = os.path.join(to_folder, name)
    print("moving: ", src, "-->", newPath)
    shutil.move(src, newPath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Split google commands train dataset.')
  parser.add_argument('--root', type=str,
                      default="speech_commands_test",
                      help='the path to the root folder of the google commands train dataset.')
  parser.add_argument('--size', type=str,
                      default="small",
                      help="size of generated dataset: 'small' or 'normal'")
  args = parser.parse_args()

  if args.size == "normal":
    assert(False, "Unimplemented!")
    remove_unused_directories(args.root, smallCategories)

    validation_files_list = os.path.join('validation_list_10cats.txt')
    test_files_list = os.path.join('testing_list_10cats.txt')

    valid_folder = os.path.join(args.root, 'valid')
    test_folder = os.path.join(args.root, 'test')
    train_folder = os.path.join(args.root, 'train')

    os.mkdir(valid_folder)
    os.mkdir(test_folder)
    os.mkdir(train_folder)

    move_files(args.root, test_folder, test_files_list)
    move_files(args.root, valid_folder, validation_files_list)
    move_directories(args.root, train_folder, smallCategories)

  elif args.size == "small":
    remove_unused_directories(args.root, smallCategories)

    validation_files_list = os.path.join('validation_list_10cats.txt')
    test_files_list = os.path.join('testing_list_10cats.txt')

    valid_folder = os.path.join(args.root, 'valid')
    test_folder = os.path.join(args.root, 'test')
    train_folder = os.path.join(args.root, 'train')

    os.mkdir(valid_folder)
    os.mkdir(test_folder)
    os.mkdir(train_folder)

    move_files(args.root, test_folder, test_files_list)
    move_files(args.root, valid_folder, validation_files_list)
    move_directories(args.root, train_folder, smallCategories)

    print("Categories not used are in 'unused' - you may want to delete them")