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
FROM python:2.7.14

# Install python dependencies
COPY requirements.txt /work/requirements.txt
WORKDIR /work
RUN pip install -r requirements.txt

# Copy and Install code
COPY . /work
RUN python setup.py develop

# Download MNIST dataset
WORKDIR /work/projects/mnist
RUN python -c 'from torchvision import datasets; \
               datasets.MNIST("data", train=True, download=True); \
               datasets.MNIST("data", train=False, download=True);'

# Download Google Speech Commands Dataset
WORKDIR /work/projects/speech_commands/data
RUN ./download_speech_commands.sh

# Experiments Root
WORKDIR /work
