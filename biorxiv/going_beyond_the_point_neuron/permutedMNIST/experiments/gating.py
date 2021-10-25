#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Experiments designed to investigate different dendritic functions that mix feedforward
and dendritic inputs. Examples include additive bias, multiplicative, multiplicative
gating, etc.
"""

from copy import deepcopy

from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    BiasingDendriticLayer,
    GatingDendriticLayer,
)
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.dendrites.mixins import EvalPerTask
from nupic.research.frameworks.vernon import mixins

from .centroid import CENTROID_10


class CentroidExperimentEvalPerTask(EvalPerTask,
                                    mixins.RezeroWeights,
                                    mixins.CentroidContext,
                                    mixins.PermutedMNISTTaskIndices,
                                    DendriteContinualLearningExperiment):
    pass


CENTROID_10_EVAL_PER_TASK = deepcopy(CENTROID_10)
CENTROID_10_EVAL_PER_TASK.update(
    experiment_class=CentroidExperimentEvalPerTask,
    tasks_to_validate=list(range(10)),
)


CENTROID_10_DENDRITE_BIAS = deepcopy(CENTROID_10_EVAL_PER_TASK)
CENTROID_10_DENDRITE_BIAS["model_args"].update(
    dendritic_layer_class=BiasingDendriticLayer
)


CENTROID_10_DENDRITE_GATE = deepcopy(CENTROID_10_EVAL_PER_TASK)
CENTROID_10_DENDRITE_GATE["model_args"].update(
    dendritic_layer_class=GatingDendriticLayer
)


CENTROID_10_DENDRITE_ABSMAXGATE = deepcopy(CENTROID_10_EVAL_PER_TASK)
CENTROID_10_DENDRITE_ABSMAXGATE["model_args"].update(
    dendritic_layer_class=AbsoluteMaxGatingDendriticLayer
)

CONFIGS = dict(
    centroid_10_eval_per_task=CENTROID_10_EVAL_PER_TASK,
    centroid_10_dendrite_bias=CENTROID_10_DENDRITE_BIAS,
    centroid_10_dendrite_absmaxgate=CENTROID_10_DENDRITE_ABSMAXGATE,
    centroid_10_dendrite_gate=CENTROID_10_DENDRITE_GATE,
    # TODO: multiplicative but not gating
)
