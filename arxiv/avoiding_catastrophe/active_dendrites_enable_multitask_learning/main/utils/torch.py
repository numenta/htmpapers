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

import torch


def convert_log_to_numpy(log):
    for k, v in log.items():
        if type(v) == torch.Tensor:
            log[k] = to_numpy(v)


def to_numpy(tensor):
    return tensor.cpu().data.numpy()


def to_tensor(var):
    return torch.Tensor(var)


def env_output_to_tensor(out):
    obs, prevrews, news, infos = out
    return tensor_or_none(obs), tensor_or_none(prevrews), tensor_or_none(news), infos


def tensor_or_none(var, device=None):
    if var is not None:
        return torch.Tensor(var, device=device)
