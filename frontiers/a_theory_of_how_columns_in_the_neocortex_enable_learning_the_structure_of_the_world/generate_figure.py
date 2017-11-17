# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

"""
Use this script to generate the figures and results presented in
(Hawkins et al., 2017). For more information plese refer to the original paper.

Jeff Hawkins, Subutai Ahmad, Yuwei Cui (2017)
A Theory of How Columns in the Neocortex Enable Learning the Structure of the
World, Frontiers in Neural Circuits 11, 81. doi:10.3389/FNCIR.2017.00081.
https://doi.org/10.3389/fncir.2017.00081
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import capacity_test
import convergence_activity
import multi_column_convergence
import ideal_classifier_experiment
import noise_convergence

RESULT_DIR_NAME = "results"
PLOTS_DIR_NAME = "plots"



def generateFigure3(cpuCount):
  """
  (A) The output layer represents each object by a sparse pattern. We tested the
  network on the first object. (B) Activity in the output layer of a single
  column network as it touches the object. The network converges after 11
  sensations (red rectangle). (C) Activity in the output layer of a three
  column network as it touches the object. The network converges much faster,
  after four sensations (red rectangle). In both (B,C) the representation in
  Column 1 is the same as the target object representation after convergence
  """
  convergence_activity.runExperiment()



def generateFigure4A(cpuCount):
  """
  Mean number of sensations needed to unambiguously recognize an object with a
  single column network as the set of learned objects increases. We train models
  on varying numbers of objects, from 1 to 100 and plot the average number of
  sensations required to unambiguously recognize a single object. The different
  curves show how convergence varies with the total number of unique features
  from which objects are constructed. In all cases the network eventually
  recognizes the object. Recognition requires fewer sensations when the set of
  features is greater
  """
  multi_column_convergence.runSingleColumnExperiment()



def generateFigure4B(cpuCount):
  """
  Mean number of observations needed to unambiguously recognize an object with
  multi-column networks as the set of columns increases. We train each network
  with 100 objects and plot the average number of sensations required to
  unambiguously recognize an object. The required number of sensations rapidly
  decreases as the number of columns increases, eventually reaching one
  """
  multi_column_convergence.runMultiColumnExperiment()


def generateFigure4C(cpuCount):
  """
  Fraction of objects that can be unambiguously recognized as a function of
  number of sensations for an ideal observer model with location (blue), without
  location (orange) and our one-column sensorimotor network (green).
  """
  ideal_classifier_experiment.single_column_accuracy_comparison()



def generateFigure5A(cpuCount):
  """
  Network capacity relative to number of mini-columns in the input layer. The
  number of output cells is kept at 4,096 with 40 cells active at any time
  """
  capacity_test.runExperiment3(cpuCount=cpuCount,
                               resultDirName=RESULT_DIR_NAME,
                               plotDirName=PLOTS_DIR_NAME)



def generateFigure5B(cpuCount):
  """
  Network capacity relative to number of cells in the output layer. The number
  of active output cells is kept at 40. The number of mini-columns in the input
  layer is 150
  """
  capacity_test.runExperiment5(cpuCount=cpuCount,
                               resultDirName=RESULT_DIR_NAME,
                               plotDirName=PLOTS_DIR_NAME)



def generateFigure5C(cpuCount):
  """
  Network capacity for one, two, and three cortical columns (CCs). The number of
  mini-columns in the input layer is 150, and the number of output cells is 4096
  """
  capacity_test.runExperiment4(cpuCount=cpuCount,
                               resultDirName=RESULT_DIR_NAME,
                               plotDirName=PLOTS_DIR_NAME)


def generateFigure6A(cpuCount):
  """
  Recognition accuracy is plotted as a function of the amount of noise in the
  sensory input (blue) and in the location input (yellow)
  """
  noise_convergence.runFeatureLocationNoiseExperiment()


def generateFigure6B(cpuCount):
  """
  Recognition accuracy as a function of the number of sensations. Colored lines
  correspond to noise levels in the location input
  """
  noise_convergence.runSensationsNoiseExperiment()


if __name__ == "__main__":

  # Map paper figures to experiment
  generateFigureFunc = {
    "3": generateFigure3,
    "4A": generateFigure4A,
    "4B": generateFigure4B,
    "4C": generateFigure4C,
    "5A": generateFigure5A,
    "5B": generateFigure5B,
    "5C": generateFigure5C,
    "6A": generateFigure6A,
    "6B": generateFigure6B,
 }
  figures = generateFigureFunc.keys()
  figures.sort()

  parser = argparse.ArgumentParser(
    description="Use this script to generate the figures and results presented in (Hawkins et al., 2017)",
    epilog="-----------------------------------------------------------------------------\
            Jeff Hawkins, Subutai Ahmad, Yuwei Cui (2017),\
            A Theory of How Columns in the Neocortex Enable Learning the Structure of the\
            World, Frontiers in Neural Circuits 11, 81. \
            https://doi.org/10.3389/fncir.2017.00081 \
            -----------------------------------------------------------------------------")

  parser.add_argument(
    "figure",
    metavar="FIGURE",
    nargs='?',
    type=str,
    default=None,
    choices=figures,
    help=("Specify the figure name to generate. Possible values are: %s " % figures)
  )
  parser.add_argument(
    "-c", "--cpuCount",
    default=None,
    type=int,
    metavar="NUM",
    help="Limit number of cpu cores.  Defaults to all available cores"
  )
  parser.add_argument(
    "-l", "--list",
    action='store_true',
    help='List all figures'
  )
  opts = parser.parse_args()

  if opts.list:
    for fig, func in sorted(generateFigureFunc.iteritems()):
      print fig, func.__doc__
  elif opts.figure is not None:
    generateFigureFunc[opts.figure](
      cpuCount=opts.cpuCount)
  else:
    parser.print_help()
