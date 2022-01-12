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

from nupic.research.frameworks.continual_learning import mixins as cl_mixins
from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    BiasingDendriticLayer,
    GatingDendriticLayer,
)
from nupic.research.frameworks.dendrites import mixins as dendrites_mixins
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.vernon import mixins as vernon_mixins

from .prototype import PROTOTYPE_10


class PrototypeExperimentEvalPerTask(dendrites_mixins.EvalPerTask,
                                     vernon_mixins.RezeroWeights,
                                     dendrites_mixins.PrototypeContext,
                                     cl_mixins.PermutedMNISTTaskIndices,
                                     DendriteContinualLearningExperiment):
    pass


PROTOTYPE_10_EVAL_PER_TASK = deepcopy(PROTOTYPE_10)
PROTOTYPE_10_EVAL_PER_TASK.update(
    experiment_class=PrototypeExperimentEvalPerTask,
    tasks_to_validate=list(range(10)),
)


PROTOTYPE_10_DENDRITE_BIAS = deepcopy(PROTOTYPE_10_EVAL_PER_TASK)
PROTOTYPE_10_DENDRITE_BIAS["model_args"].update(
    dendritic_layer_class=BiasingDendriticLayer
)


PROTOTYPE_10_DENDRITE_GATE = deepcopy(PROTOTYPE_10_EVAL_PER_TASK)
PROTOTYPE_10_DENDRITE_GATE["model_args"].update(
    dendritic_layer_class=GatingDendriticLayer
)


PROTOTYPE_10_DENDRITE_ABSMAXGATE = deepcopy(PROTOTYPE_10_EVAL_PER_TASK)
PROTOTYPE_10_DENDRITE_ABSMAXGATE["model_args"].update(
    dendritic_layer_class=AbsoluteMaxGatingDendriticLayer
)

CONFIGS = dict(
    prototype_10_eval_per_task=PROTOTYPE_10_EVAL_PER_TASK,
    prototype_10_dendrite_bias=PROTOTYPE_10_DENDRITE_BIAS,
    prototype_10_dendrite_absmaxgate=PROTOTYPE_10_DENDRITE_ABSMAXGATE,
    prototype_10_dendrite_gate=PROTOTYPE_10_DENDRITE_GATE,
    # TODO: multiplicative but not gating
)
