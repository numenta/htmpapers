# -*- coding: utf-8 -*-
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
from __future__ import print_function
import math
import os
from tabulate import tabulate

from expsuite import PyExperimentSuite

# Constants values used across all experiments
STRIDE = 1
PADDING = 0
KERNEL_SIZE = 5

def computeMaxPool(input_width):
  """
  Compute CNN max pool width
  """
  wout = math.floor((input_width + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1)
  return int(math.floor(wout / 2.0))

if __name__ == '__main__':
  suite = PyExperimentSuite()
  suite.parse_opt()
  suite.parse_cfg()
  experiments = suite.options.experiments or suite.cfgparser.sections()

  paramsTable = [["Network",
                  "L1 F", "L1 Sparsity",
                  "L2 F", "L2 Sparsity",
                  "L3 N", "L3 Sparsity",
                  "Wt Sparsity"]]
  for name in experiments:

    # Iterate over experiments, skipping over errors.
    try:
      exps = suite.get_exps(suite.get_exp(name)[0])
    except:
      print("Couldn't parse experiment:", name)
      continue

    for exp in exps:
      if not os.path.exists(exp):
        continue
      params = suite.get_params(exp=exp)

      l3_n = params["n"]
      l3_k = params["k"]
      l3_sp = "{0:.1f}%".format(100 * float(l3_k) / l3_n)
      wt_sp = "{0}%".format(100 * float(params["weight_sparsity"]))

      c1_k = params["c1_k"]
      if isinstance(c1_k, basestring):
        c1_k = map(int, c1_k.split("_"))
        l1_k = c1_k[0]
        l2_k = c1_k[1]
      else:
        l1_k = int(c1_k)
        l2_k = None

      c1_out_channels = params["c1_out_channels"]
      inputFeatures = map(int, params["c1_input_shape"].split("_"))
      if isinstance(c1_out_channels, basestring):
        c1_out_channels = map(int, c1_out_channels.split("_"))

        # Compute CNN output len
        l1_maxpoolWidth = computeMaxPool(inputFeatures[2])
        l1_f = c1_out_channels[0]
        l1_len = int(l1_maxpoolWidth * l1_maxpoolWidth * l1_f)
        l1_sp = "{0:.1f}%".format(100 * float(l1_k) / l1_len)

        # Feed CNN-1 output to CNN-2
        l2_maxpoolWidth = computeMaxPool(l1_maxpoolWidth)
        l2_f = c1_out_channels[1]
        l2_len = int(l2_maxpoolWidth * l2_maxpoolWidth * l2_f)
        l2_sp = "{0:.1f}%".format(100 * float(l2_k) / l2_len)
      else:
        # Compute CNN output len
        l1_maxpoolWidth = computeMaxPool(inputFeatures[2])
        l1_f = int(c1_out_channels)
        l1_len = int(l1_maxpoolWidth * l1_maxpoolWidth * l1_f)
        l1_sp = "{0:.1f}%".format(100 * float(l1_k) / l1_len)

        l2_f = None
        l2_sp = None

      paramsTable.append([name, l1_f, l1_sp, l2_f, l2_sp, l3_n, l3_sp, wt_sp])

  print()
  print(tabulate(paramsTable, headers="firstrow", tablefmt="grid"))
  print()
  print(tabulate(paramsTable, headers="firstrow", tablefmt="latex"))
