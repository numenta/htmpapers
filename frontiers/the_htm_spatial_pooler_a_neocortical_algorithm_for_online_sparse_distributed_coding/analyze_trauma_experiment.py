# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
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

from optparse import OptionParser
from htmresearch.support.sp_paper_utils import *
import matplotlib as mpl
from nupic.math.topology import indexFromCoordinates
import nupic.math.topology as topology

mpl.rcParams['pdf.fonttype'] = 42

"""
Script to analyze trauma experiment result
"""

def getInputConverage(expName, numEpochs, killAt):
  inputCoverageTraumaRegion = []
  inputCoverageControlRegion = []
  inputCoverageBeforeTrauma = None
  inputCoverageAfterTrauma = None

  epochList = []
  for epoch in range(99, numEpochs):
    epochList.append(epoch)
    inputSpaceCoverage = np.load('results/InputCoverage/{}/epoch_{}.npz'.format(expName, epoch))
    inputSpaceCoverage = inputSpaceCoverage['arr_0']
    inputCoverageTraumaRegion.append(
      np.mean(inputSpaceCoverage[14:18, 14:18]))

    inputCoverageControlRegion.append(
      np.mean(inputSpaceCoverage[27:31, 8:12]))

    if epoch == killAt - 1:
      inputCoverageBeforeTrauma = inputSpaceCoverage
    if epoch == killAt:
      inputCoverageAfterTrauma = inputSpaceCoverage

  inputSpaceCoverage = np.load \
    ('results/InputCoverage/{}/epoch_{}.npz'.format(expName, epoch))
  inputCoverageAfterRecovery = inputSpaceCoverage['arr_0']
  return (epochList,
          inputCoverageTraumaRegion,
          inputCoverageControlRegion,
          inputCoverageBeforeTrauma,
          inputCoverageAfterTrauma,
          inputCoverageAfterRecovery)


def plotInputCoverage(expName, epochList, killAt,
                      inputCoverageControlRegion,
                      inputCoverageTraumaRegion, ylim):
  # Plot coverage factor for trauma region and a control region
  fig, axs = plt.subplots(2, 2)
  axs[0, 0].plot(epochList, inputCoverageControlRegion, label='control')
  axs[0, 0].plot(epochList, inputCoverageTraumaRegion, label='trauma')
  axs[0, 0].plot([killAt, killAt], ylim, 'k--')
  # axs[0, 0].set_xlim([killAt-50, numEpochs])
  axs[0, 0].set_xlim([100, 500])
  # axs[0, 0].set_ylim([19, 50])
  # axs[0, 0].set_ylim([-1, 50])
  axs[0, 0].set_ylim(ylim)
  axs[0, 0].set_xlabel('Time')
  # axs[0, 0].set_aspect('equal')
  plt.axis('equal')
  plt.legend()
  plt.savefig('figures/traumaRecovery_{}.pdf'.format(expName))


def plotSynapseGrowth(expName,
                      inputCoverageAfterRecovery,
                      inputCoverageAfterTrauma):
  # Compare before and after recovery
  fig, axs = plt.subplots(2, 2)
  im = axs[0, 0].pcolor(
    inputCoverageAfterRecovery.astype('float32') - inputCoverageAfterTrauma,
  vmin=-12, vmax=12)
  axs[0, 0].set_title('Synapse Growth')
  axs[0, 0].set_xlim([-1, 33])
  axs[0, 0].set_ylim([-1, 33])
  axs[0, 0].set_aspect('equal')
  plt.axis('equal')
  cax = fig.add_axes([0.15, 0.4, 0.3, 0.05])
  fig.colorbar(im, cax=cax, orientation='horizontal',
               ticks=[-10, -5, 0, 5, 10])
  plt.savefig('figures/synapseGrowthAfterTrauma_{}.pdf'.format(expName))



def plotMovie(expName, numEpochs, killAt):
  print " create frames for the movie"
  # plot RFcenters and inputCoverage over training
  centerColumn = indexFromCoordinates((15, 15), (32, 32))
  deadCols = topology.wrappingNeighborhood(centerColumn, 5, (32, 32))

  for epoch in range(1, numEpochs):
    print "processing epoch {}/{} ".format(epoch, numEpochs)
    fig, axs = plt.subplots(2, 2)
    RFcenterInfo = np.load('results/RFcenters/{}/epoch_{}.npz'.format(expName, epoch))
    RFcenters = RFcenterInfo['arr_0']
    avgDistToCenter = RFcenterInfo['arr_1']
    axs[0, 1].scatter(RFcenters[:, 0], RFcenters[:, 1], s=4, c='black')
    axs[0, 1].set_title('RF centers')
    axs[0, 1].set_xlim([-1, 33])
    axs[0, 1].set_ylim([-1, 33])
    axs[0, 1].set_aspect('equal')
    plt.axis('equal')

    # # plot input space coverage
    inputSpaceCoverage = np.load('results/InputCoverage/{}/epoch_{}.npz'.format(expName, epoch))
    inputSpaceCoverage = inputSpaceCoverage['arr_0']
    im1 = axs[0, 0].pcolor(inputSpaceCoverage, vmin=10, vmax=80)
    axs[0, 0].set_title('Input Space Coverage')
    axs[0, 0].set_xlim([-1, 33])
    axs[0, 0].set_ylim([-1, 33])
    axs[0, 0].set_aspect('equal')
    plt.axis('equal')

    boostFactors = np.load('results/boostFactors/{}/epoch_{}.npz'.format(expName, epoch))
    boostFactors = boostFactors['arr_0']
    # boostFactors[deadCols] = np.nan
    boostFactors = np.reshape(boostFactors, (32, 32))
    im2 = axs[1, 0].pcolor(boostFactors, vmin=0.5, vmax=1.5)
    axs[1, 0].set_title('Boost Factors')
    axs[1, 0].set_xlim([-1, 33])
    axs[1, 0].set_ylim([-1, 33])
    axs[1, 0].set_aspect('equal')
    plt.axis('equal')

    # fig.delaxes(axs[1, 0])
    cax = fig.add_axes([0.55, 0.4, 0.3, 0.05])
    fig.colorbar(im1, cax=cax, orientation='horizontal',
                 ticks=[0, 20, 40, 60, 80])
    cax.set_xlabel('input space coverage')

    cax = fig.add_axes([0.55, 0.2, 0.3, 0.05])
    fig.colorbar(im2, cax=cax, orientation='horizontal',
                 ticks=[0, .5, 1, 1.5, 2])
    cax.set_xlabel('boost factors')

    fig.delaxes(axs[1, 1])

    plt.savefig('figures/traumaMovie/{}_frame_{}.png'.format(expName, epoch))
    plt.close(fig)


def _getArgs():
  parser = OptionParser(usage="%prog -e experiment_name"
                        )
  parser.add_option("-e",
                    "--expName",
                    type=str,
                    default='trauma_inputs_with_topology_seed_41',
                    dest="expName",
                    help="experiment Name")

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder


if __name__ == "__main__":
  (_options, _args) = _getArgs()
  expName = _options.expName
  plt.ion()
  # expName = 'trauma_boosting_with_topology_seed_41'
  # expName = 'trauma_inputs_with_topology_seed_41'
  numEpochs = 500
  killAt = 180

  if "inputs" in expName:
    ylim = [-1, 50]
  else:
    ylim = [19, 50]

  (epochList,
   inputCoverageTraumaRegion,
   inputCoverageControlRegion,
   inputCoverageBeforeTrauma,
   inputCoverageAfterTrauma,
   inputCoverageAfterRecovery) = getInputConverage(expName, numEpochs, killAt)

  plotInputCoverage(expName, epochList, killAt,
                    inputCoverageControlRegion,
                    inputCoverageTraumaRegion, ylim)

  plotSynapseGrowth(expName,
                    inputCoverageAfterRecovery,
                    inputCoverageAfterTrauma)

  plotMovie(expName, numEpochs, killAt)

