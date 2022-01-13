# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

from .base import singleseg_mt10_base 

from copy import deepcopy

'''
Round 1 of experiments
'''

baseline_1d = deepcopy(singleseg_mt10_base)
baseline_1d.update(
    policy_lr=3.0e-4,
    qf_lr=3.0e-4,
    hidden_sizes=(520, 520),
    kw_percent_on=0.25,
    fp16=True,
    weight_sparsity=0.5,
    preprocess_output_dim=32,
)


CONFIGS = dict(
    baseline_1d=baseline_1d,
)