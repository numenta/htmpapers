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
Use this script to generate the figures and results presented in (Hawkins et al., 2017).
For more information plese refer to the original paper.

Jeff Hawkins, Subutai Ahmad, Yuwei Cui (2017)
Why Does the Neocortex Have Layers and Columns, a Theory of Learning the  
3D Structure of the World, Preprint of journal submission. doi: 10.1101/162263.
http://www.biorxiv.org/content/early/2017/07/12/162263
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import capacity_test
import convergence_activity
import multi_column_convergence

RESULT_DIR_NAME = "results"
PLOTS_DIR_NAME = "plots"



def generateFigure4(cpuCount):
  """
  illustrates the rate of convergence for a one column network(4B) and for a
  three-column network(4C).
  """
  convergence_activity.runExperiment()



def generateFigure5A(cpuCount):
  """
  Mean number of observations to unambiguously recognize an object with a single 
  column network as the set of learned objects increases. We train models on 
  varying numbers of objects, from 1 to 100 and plot the average number of 
  touches required to unambiguously recognize a single object. The different 
  curves show how convergence varies with the total number of unique features 
  from which objects are constructed. In all cases the network is able to 
  eventually recognize the object, but the recognition is much faster when the 
  set of features is greater
  """
  multi_column_convergence.runExperiment5()



def generateFigure5B(cpuCount):
  """
  Mean number of observations to unambiguously recognize an object with 
  multi-column networks as the set of learned objects increases. We train each 
  network with 100 objects and plot the average number of touches required to 
  unambiguously recognize an object. Recognition time improves rapidly as the 
  number of columns increases
  """
  multi_column_convergence.runExperiment3()



def generateFigure6A(cpuCount):
  """
  Network capacity relative to number of minicolumns in the input layer. The 
  number of output cells is kept at 4096 with 40 cells active at any time  
  """
  capacity_test.runExperiment3(cpuCount=cpuCount,
                               resultDirName=RESULT_DIR_NAME,
                               plotDirName=PLOTS_DIR_NAME)



def generateFigure6B(cpuCount):
  """
  Network capacity relative to number of cells in the output layer. The number 
  of active output cells is kept  at 40. The number of minicolumns in the input 
  layer is 150  
  """
  capacity_test.runExperiment5(cpuCount=cpuCount,
                               resultDirName=RESULT_DIR_NAME,
                               plotDirName=PLOTS_DIR_NAME)



def generateFigure6C(cpuCount):
  """
  Network capacity for one, two, and three cortical columns (CCs). The number of 
  minicolumns in the input layer is 150, and the number of output cells is 4096  
  """
  capacity_test.runExperiment4(cpuCount=cpuCount,
                               resultDirName=RESULT_DIR_NAME,
                               plotDirName=PLOTS_DIR_NAME)



if __name__ == "__main__":

  # Map paper figures to experiment
  generateFigureFunc = {
    "4": generateFigure4,
    "5A": generateFigure5A,
    "5B": generateFigure5B,
    "6A": generateFigure6A,
    "6B": generateFigure6B,
    "6C": generateFigure6C
  }
  figures = generateFigureFunc.keys()
  figures.sort()

  parser = argparse.ArgumentParser(
    description="Use this script to generate the figures and results presented in (Hawkins et al., 2017)",
    epilog="-----------------------------------------------------------------------------\
            Jeff Hawkins, Subutai Ahmad, Yuwei Cui (2017)\
            Why Does the Neocortex Have Layers and Columns, a Theory of Learning the  \
            3D Structure of the World, Preprint of journal submission. doi: 10.1101/162263.\
            http://www.biorxiv.org/content/early/2017/07/12/162263\
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
