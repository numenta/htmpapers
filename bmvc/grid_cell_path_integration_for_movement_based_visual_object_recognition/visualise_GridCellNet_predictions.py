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
Uses a pretrained decoder and the SDRs predicted by a GridCellNet classifier
to visualise the networks representations.
Note that up until inference takes place, representations include sensations
received by the network in previous time-steps (i.e. "ground-truth" features),
but after inference, predicted SDR features are only based on the network's
own representation.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from SDR_decoder import MLPDecoder

torch.manual_seed(18)
np.random.seed(18)

DATASET = "mnist"


def predict_column_plus_reps(net, prediction_sequence, touch_sequence, label,
                             num_sensations_to_inference, ground_truth):
    """
    Output images corresponding to the network's representation as it progressively
    senses the input and makes predictions about the next sensation. Until inference
    has taken place, the representation consists of all previously sensed features,
    and a prediction of the next step. Once inference has taken place, all additional
    features in the representation are based on predictions.
    :param prediction_sequence : contains the predictions made by the network as well
    as previously sensed features
    :param touch_sequence : contains the order of the networks sensations, which is
    needed to match the predicted features to their correct spatial locations before
    feeding to the decoder
    """
    plt.imsave("predicted_images/" + label + "_ground_truth.png", ground_truth)

    for touch_iter in range(len(prediction_sequence)):

        input_sdr = np.zeros([128, 5 * 5])
        current_sequence = prediction_sequence[touch_iter]

        for sequence_iter in range(len(current_sequence)):
            if len(current_sequence[sequence_iter]) > 0:
                input_sdr[current_sequence[sequence_iter],
                          touch_sequence[sequence_iter]] = 1

        input_sdr = torch.from_numpy(np.reshape(input_sdr, 128 * 5 * 5))
        input_sdr = input_sdr.type(torch.DoubleTensor)

        reconstructed = net(input_sdr)

        # Highlight the location of where the current prediction is taking place
        # Note that as going we're from a 5*5 feature space to a 28*28 space, this
        # is only approximate
        current_touch = touch_sequence[touch_iter]
        width_iter = current_touch // 5
        height_iter = current_touch % 5
        highlight_width_lower, highlight_width_upper = ((1 + width_iter * 5),
                                                        (1 + (width_iter + 1) * 5))
        highlight_height_lower, highlight_height_upper = ((1 + height_iter * 5),
                                                          (1 + (height_iter + 1) * 5))

        highlight_array = np.zeros((28, 28))
        highlight_array[highlight_width_lower:highlight_width_upper,
                        highlight_height_lower:highlight_height_upper] = 0.5

        if num_sensations_to_inference is not None:
            if touch_iter >= num_sensations_to_inference:
                # Add highlight to borders to indicate inference successful,
                # and that all future representations are based on model predictions
                highlight_array[0, :] = 1.0
                highlight_array[27, :] = 1.0
                highlight_array[:, 0] = 1.0
                highlight_array[:, 27] = 1.0

        reconstructed = np.clip(reconstructed.detach().numpy() + highlight_array, 0, 1)

        if num_sensations_to_inference is not None:
            prediction = "correctly_classified"
        else:
            prediction = "misclassified"

        plt.imsave("predicted_images/" + label + "_" + prediction
                   + "_touch_" + str(touch_iter) + ".png", reconstructed[0, :, :])


if __name__ == "__main__":

    if os.path.exists("predicted_images/") is False:
        try:
            os.mkdir("predicted_images/")
        except OSError:
            pass

    object_prediction_sequences = np.load(
        "python2_htm_docker/docker_dir/prediction_data/"
        "object_prediction_sequences.npy", allow_pickle=True, encoding="latin1")

    net = MLPDecoder().double()
    net.load_state_dict(torch.load("saved_networks/" + DATASET + "_decoder.pt"))

    print("Visualising predictions from " + str(len(object_prediction_sequences))
          + " objects")

    for object_iter in range(len(object_prediction_sequences)):

        current_object = object_prediction_sequences[object_iter]

        predict_column_plus_reps(
            net, prediction_sequence=current_object["prediction_sequence"],
            touch_sequence=current_object["touch_sequence"],
            label=current_object["name"],
            num_sensations_to_inference=current_object["numSensationsToInference"],
            ground_truth=current_object["ground_truth_image"])
