#!/usr/bin/env sh
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
set -e

FILE_NAME=speech_commands_v0.01.tar.gz
URL=http://download.tensorflow.org/data/$FILE_NAME
echo "downloading $URL...\n"
wget $URL

DATASET_FOLDER=speech_commands
echo "extracting $FILE_NAME..."
mkdir -p $DATASET_FOLDER
tar -xzf $FILE_NAME -C $DATASET_FOLDER

echo "splitting the dataset into train, validation and test sets..."
python split_dataset.py --root=$DATASET_FOLDER --size=small

echo "done"
