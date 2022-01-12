# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------

# from: https://github.com/openai/baselines/blob/master/baselines/common/mpi_moments.py

import torch


def mpi_mean(x, axis=0, comm=None, keepdims=False):
    assert x.dim() > 0
    xsum = x.sum(axis=axis, keepdim=keepdims)
    n = xsum.numel()
    localsum = torch.zeros(n + 1, dtype=x.dtype)
    localsum[:n] = xsum.flatten()  # TODO: switch to ravel if pytorch >= 1.9
    localsum[n] = x.shape[axis]
    globalsum = torch.zeros_like(localsum)
    globalsum = localsum
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]


def mpi_moments(x, axis=0, comm=None, keepdims=False):
    assert x.dim() > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = torch.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = torch.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis + 1 :]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count
