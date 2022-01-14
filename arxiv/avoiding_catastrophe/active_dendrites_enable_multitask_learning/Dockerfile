# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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
FROM ubuntu:18.04

# Install OS dependencies required to install mujoco_py on ubuntu
RUN apt update -q \
    && apt install -y \
    curl \
    wget \
    unzip \ 
    git \
    build-essential \
    patchelf \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libjpeg-dev \
    libffi-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /root/miniconda \
    && rm /tmp/miniconda.sh

# Install MuJoCo binaries and activation key
RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O /tmp/mujoco.zip \
    && mkdir -p /root/.mujoco \
    && unzip /tmp/mujoco.zip -d /root/.mujoco/ \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm /tmp/mujoco.zip
RUN wget https://www.roboti.us/file/mjkey.txt -O /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

# Create conda enviroment preconfigured to run multi-task learning experiments
COPY . /work
WORKDIR /work
RUN export PATH=/root/miniconda/bin:$PATH \
    && export LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH} \
    && ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so \
    && git config --global http.postBuffer 524288000 \
    && conda env create \
    && echo "source activate multitask_dendrites" >> ~/.bashrc
ENV PATH /root/miniconda/bin:${PATH}

# Environment variables used by the experiments. Use "-e" docker option to override
ENV CHECKPOINT_DIR /checkpoints
ENV WANDB_DIR /wandb
