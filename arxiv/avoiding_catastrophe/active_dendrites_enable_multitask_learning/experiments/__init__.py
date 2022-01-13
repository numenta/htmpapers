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
from .sampler import CONFIGS as SAMPLER
from .singleseg_experiments import CONFIGS as SINGLESEGKW
from .multiseg_experiments import CONFIGS as MULTISEGKW
from .mlp_experiments import CONFIGS as MLP
from .paper_experiments import CONFIGS as PAPER
from .hooks import CONFIGS as HOOKS

"""
Import and collect all experiment configurations into one CONFIG
"""
__all__ = ["CONFIGS"]

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(BASE)
CONFIGS.update(SAMPLER)
CONFIGS.update(SINGLESEGKW)
CONFIGS.update(MULTISEGKW)
CONFIGS.update(MLP)
CONFIGS.update(HOOKS)
CONFIGS.update(PAPER)