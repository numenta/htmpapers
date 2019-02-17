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
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import torch



def _nonzero_counter_hook(module, inputs, output):
  """
  Module hook used to count the number of nonzero floating point values from
  all the tensors used by the given network during inference. This hook will be
  called every time before :func:`forward` is invoked.

  See :func:`torch.nn.Module.register_forward_hook`
  """
  if not hasattr(module, "__counter_nonzero__"):
    raise ValueError("register_counter_nonzero was not called for this network")

  if module.training:
    return

  size = module.__counter_nonzero__.get("input", 0)
  size += sum([torch.nonzero(i).size(0) for i in inputs])
  module.__counter_nonzero__["input"] = size

  size = module.__counter_nonzero__.get("output", 0)
  size += torch.nonzero(output).size(0)
  module.__counter_nonzero__["output"] = size

  for name, param in module._parameters.items():
    if param is None:
      continue

    size = module.__counter_nonzero__.get(name, 0)
    size += torch.nonzero(param.data).size(0)
    module.__counter_nonzero__[name] = size

  for name, buffer in module._buffers.items():
    if buffer is None:
      continue

    size = module.__counter_nonzero__.get(name, 0)
    size += torch.nonzero(buffer).size(0)
    module.__counter_nonzero__[name] = size



def register_nonzero_counter(network, stats):
  """
  Register forward hooks to count the number of nonzero floating points
  values from all the tensors used by the given network during inference.

  :param network: The network to attach the counter
  :param stats: Dictionary holding the counter.
  """
  if hasattr(network, "__counter_nonzero__"):
    raise ValueError("nonzero counter was already registered for this network")

  if not isinstance(stats, dict):
    raise ValueError("stats must be a dictionary")

  network.__counter_nonzero__ = stats
  handles = []
  for name, module in network.named_modules():
    handles.append(module.register_forward_hook(_nonzero_counter_hook))
    if network != module:
      if hasattr(module, "__counter_nonzero__"):
        raise ValueError("nonzero counter was already registered for this module")

      child_data = dict()
      network.__counter_nonzero__[name] = child_data
      module.__counter_nonzero__ = child_data

  network.__counter_nonzero_handles__ = handles



def unregister_counter_nonzero(network):
  """
  Unregister nonzero counter hooks
  :param network: The network previously registered via `register_nonzero_counter`
  """
  if not hasattr(network, "__counter_nonzero_handles__"):
    raise ValueError("register_counter_nonzero was not called for this network")

  for h in network.__counter_nonzero_handles__:
    h.remove()

  delattr(network, "__counter_nonzero_handles__")
  for module in network.modules():
    if hasattr(module, "__counter_nonzero__"):
      delattr(module, "__counter_nonzero__")
