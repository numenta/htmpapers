#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Calculate accuracy of GridCellNet as a function of the number of sensations
Can be used to compare performance with k-NN by using SDR_classifiers.py with
KNN_PROGRESSIVE_SENSATIONS_BOOL=True
"""

import matplotlib.pyplot as plt
import numpy as np

INPUT_GRID_DIMENSION = 5


def plot_sensations_to_inference(object_prediction_sequences):

    num_sensations_list = []

    for object_iter in range(len(object_prediction_sequences)):

        num_sensations_list.append(
            object_prediction_sequences[object_iter]["numSensationsToInference"])

    cumm_percent_inferred = []
    num_correct = 0

    for num_sensation_iter in range(INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION):

        num_correct += len(
            np.nonzero(np.array(num_sensations_list) == (num_sensation_iter + 1))[0])

        cumm_percent_inferred.append(num_correct
                                     / len(object_prediction_sequences))

    plt.scatter(list(range(1, INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION + 1)),
                cumm_percent_inferred)
    plt.ylim(0, 1)
    plt.show()

    print("Cummulative percent inferred as a function of the number of sensations")
    print(cumm_percent_inferred)

    return cumm_percent_inferred


if __name__ == "__main__":

    object_prediction_sequences = np.load(
        "python2_htm_docker/docker_dir/prediction_data/"
        "object_prediction_sequences.npy", allow_pickle=True,
        encoding="latin1")

    plot_sensations_to_inference(object_prediction_sequences)
