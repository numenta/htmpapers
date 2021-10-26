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

from .base import CONFIGS as BASE
from .batch import CONFIGS as BATCH
from .batch_mnist import CONFIGS as BATCH_MNIST
from .centroid import CONFIGS as CENTROID
from .centroid_ten_segments import CONFIGS as CENTROID_TEN_SEGMENTS
from .dendrites_xor_sparsity import CONFIGS as DENDRITES_XOR_SPARSITY
from .gating import CONFIGS as GATING
from .hyperparameter_search import CONFIGS as HYPERPARAMETERSEARCH
from .mlp import CONFIGS as MLP
from .no_dendrites import CONFIGS as NO_DENDRITES
from .profiler import CONFIGS as PROFILER
from .si_centroid import CONFIGS as SI_CENTROID
from .sp_context import CONFIGS as SP_CONTEXT
from .sp_context_search import CONFIGS as SP_PROTO

"""
Import and collect all experiment configurations into one CONFIG
"""
__all__ = ["CONFIGS"]

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(BASE)
CONFIGS.update(BATCH)
CONFIGS.update(BATCH_MNIST)
CONFIGS.update(CENTROID)
CONFIGS.update(CENTROID_TEN_SEGMENTS)
CONFIGS.update(DENDRITES_XOR_SPARSITY)
CONFIGS.update(GATING)
CONFIGS.update(HYPERPARAMETERSEARCH)
CONFIGS.update(MLP)
CONFIGS.update(NO_DENDRITES)
CONFIGS.update(PROFILER)
CONFIGS.update(SI_CENTROID)
CONFIGS.update(SP_CONTEXT)
CONFIGS.update(SP_PROTO)
