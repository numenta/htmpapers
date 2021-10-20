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
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from auto_encoder_sequential import PatchDecoder
from SDR_decoder import MLPDecoder

seed_val = 1
torch.manual_seed(seed_val)
np.random.seed(seed_val)

DECODER_ARCHITECTURE = "abstract_SDR_decoder"  # Options are "patch_decoder" (for
# 7x7 pixel patches) or "abstract_SDR_decoder" (for the larger, CNN based patch
# predictions)
DATASET = "mnist"
INPUT_GRID_DIMENSION = 5


def predict_GridCellNet_small_patch_reps(net, prediction_sequence,  # noqa: N802
                                         touch_sequence,
                                         label, num_sensations_to_converge,
                                         ground_truth):
    """
    Use predictions from the network trained on separate 7x7 pixel patches along a
    4x4 grid space
    """
    plt.imsave("predicted_images/" + label + "_ground_truth.png", ground_truth)

    contains_empty_predictions = 0

    for touch_iter in range(len(prediction_sequence)):

        input_sdr = np.zeros([128, 4, 4])

        current_sequence = prediction_sequence[touch_iter]

        for sequence_iter in range(len(current_sequence)):

            # Track any empty predictions after convergence to a single representation
            # This can occur due to e.g. noisy alignment of grid cell representations
            # and learned weights
            if len(current_sequence[sequence_iter]) == 0:

                if sequence_iter >= num_sensations_to_converge:

                    contains_empty_predictions = 1

            if len(current_sequence[sequence_iter]) > 0:

                x_ref = touch_sequence[sequence_iter] // INPUT_GRID_DIMENSION
                y_ref = touch_sequence[sequence_iter] % INPUT_GRID_DIMENSION

                input_sdr[current_sequence[sequence_iter],
                          x_ref, y_ref] = 1

        image_reconstruct = np.zeros((28, 28))

        # Go through the generated SDR and create the pixel-based output using the
        # decoder NB where a sensation or prediction has not yet been made, a
        # zero-vector is fed to the decoder
        for patch_width_iter in range(4):

            for patch_height_iter in range(4):

                current_input_sdr = torch.from_numpy(input_sdr[:, patch_width_iter,
                                                               patch_height_iter])

                image_reconstruct[patch_width_iter * 7:(patch_width_iter + 1) * 7,
                                  patch_height_iter
                                  * 7:(patch_height_iter + 1)
                                  * 7] = net(current_input_sdr).detach().numpy()

        # Create a bright border to indicate when convergence successful,
        # and that all future representations are based on model predictions
        border_array = np.zeros((28, 28))

        if num_sensations_to_converge is not None:
            if touch_iter >= num_sensations_to_converge:
                border_array[0, :] = 1.0
                border_array[27, :] = 1.0
                border_array[:, 0] = 1.0
                border_array[:, 27] = 1.0

        image_reconstruct = np.clip(image_reconstruct + border_array, 0, 1)

        # Add four indiicator pixels to emphasize where the current prediction
        # is taking place
        current_touch = touch_sequence[touch_iter]
        width_iter = current_touch // 4
        height_iter = current_touch % 4
        highlight_width_lower, highlight_width_upper = ((width_iter * 7),
                                                        ((width_iter + 1) * 7 - 1))
        highlight_height_lower, highlight_height_upper = ((height_iter * 7),
                                                          ((height_iter + 1) * 7 - 1))

        # Convert corner pixels to a bright pixel (if currently dark), or a dark
        # pixel if currently bright, signifying where the current prediction is
        image_reconstruct[highlight_width_lower,
                          highlight_height_lower] = float(image_reconstruct[
                                                          highlight_width_lower,
                                                          highlight_height_lower]
                                                          < 0.8)

        image_reconstruct[highlight_width_upper,
                          highlight_height_upper] = float(image_reconstruct[
                                                          highlight_width_upper,
                                                          highlight_height_upper]
                                                          < 0.8)

        image_reconstruct[highlight_width_upper,
                          highlight_height_lower] = float(image_reconstruct[
                                                          highlight_width_upper,
                                                          highlight_height_lower]
                                                          < 0.8)

        image_reconstruct[highlight_width_lower,
                          highlight_height_upper] = float(image_reconstruct[
                                                          highlight_width_lower,
                                                          highlight_height_upper]
                                                          < 0.8)

        plt.imsave("predicted_images/" + label + "_recondstruction__touch_"
                   + str(touch_iter) + ".png", image_reconstruct)
        plt.imsave("predicted_images/" + label + "_ground_truth.png", ground_truth)

    return contains_empty_predictions


def predict_GridCellNet_broad_patch_reps(net, prediction_sequence,  # noqa: N802
                                         touch_sequence,
                                         label, num_sensations_to_converge,
                                         ground_truth):
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

    contains_empty_predictions = 0

    for touch_iter in range(len(prediction_sequence)):

        input_sdr = np.zeros([128, 5 * 5])
        random_input_sdr = np.zeros([128, 5 * 5])

        current_sequence = prediction_sequence[touch_iter]

        for sequence_iter in range(len(current_sequence)):

            # Track any empty predictions after convergence to a single representation
            # This can occur due to e.g. noisy alignment of grid cell representations
            # and learned weights
            if len(current_sequence[sequence_iter]) == 0:

                if sequence_iter >= num_sensations_to_converge:

                    contains_empty_predictions = 1

            if len(current_sequence[sequence_iter]) > 0:

                input_sdr[current_sequence[sequence_iter],
                          touch_sequence[sequence_iter]] = 1

                if sequence_iter >= num_sensations_to_converge:
                    # Create a complementary control representation where
                    # predicted representations are replaced by random SDRs
                    # NB this is not done for the predictions made before
                    # convergence, but only for predictions that take place
                    # at the point of convergence and onward

                    random_indices = random.sample(
                        range(128),
                        len(current_sequence[sequence_iter]))

                    random_input_sdr[random_indices,
                                     touch_sequence[sequence_iter]] = 1

                else:
                    # If not yet converged, use the stored representation
                    random_input_sdr[current_sequence[sequence_iter],
                                     touch_sequence[sequence_iter]] = 1

        input_sdr = torch.from_numpy(np.reshape(input_sdr, 128 * 5 * 5))
        input_sdr = input_sdr.type(torch.DoubleTensor)

        random_input_sdr = torch.from_numpy(np.reshape(random_input_sdr, 128 * 5 * 5))
        random_input_sdr = random_input_sdr.type(torch.DoubleTensor)

        reconstructed = net(input_sdr)
        random_reconstructed = net(random_input_sdr)

        # Highlight the location of where the current prediction is taking place
        # Note that as we're going from a 5*5 feature space to a 28*28 space, this
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

        if num_sensations_to_converge is not None:
            if touch_iter >= num_sensations_to_converge:
                # Add highlight to borders to indicate convergence successful,
                # and that all future representations are based on model predictions
                highlight_array[0, :] = 1.0
                highlight_array[27, :] = 1.0
                highlight_array[:, 0] = 1.0
                highlight_array[:, 27] = 1.0

        reconstructed = np.clip(reconstructed.detach().numpy() + highlight_array, 0, 1)
        random_reconstructed = np.clip(random_reconstructed.detach().numpy()
                                       + highlight_array, 0, 1)

        if num_sensations_to_converge is not None:
            prediction = "correctly_classified"

        plt.imsave("predicted_images/" + label + "_" + prediction
                   + "_touch_" + str(touch_iter) + ".png", reconstructed[0, :, :])

        plt.imsave("predicted_images/" + label + "_random_control_touch_"
                   + str(touch_iter) + ".png",
                   random_reconstructed[0, :, :])

    return contains_empty_predictions


if __name__ == "__main__":

    if os.path.exists("predicted_images/") is False:
        try:
            os.mkdir("predicted_images/")
        except OSError:
            pass

    object_prediction_sequences = np.load(
        "python2_htm_docker/docker_dir/prediction_data/"
        "object_prediction_sequences.npy", allow_pickle=True, encoding="latin1")

    if DECODER_ARCHITECTURE == "patch_decoder":

        net = PatchDecoder().double()
        net.load_state_dict(torch.load("saved_networks/"
                                       + DATASET + "_patch_decoder.pt"))
        net.eval()

    elif DECODER_ARCHITECTURE == "abstract_SDR_decoder":
        net = MLPDecoder().double()
        net.load_state_dict(torch.load("saved_networks/" + DATASET + "_decoder.pt"))
        net.eval()

    print("Visualising predictions from " + str(len(object_prediction_sequences))
          + " potential objects")

    objects_with_convergence = 0
    objects_with_empty_predictions = 0

    for object_iter in range(len(object_prediction_sequences)):

        current_object = object_prediction_sequences[object_iter]

        if current_object["numSensationsToSingleConverge"] is not None:

            if DECODER_ARCHITECTURE == "patch_decoder":

                contains_empty_predictions = predict_GridCellNet_small_patch_reps(
                    net, prediction_sequence=current_object["prediction_sequence"],
                    touch_sequence=current_object["touch_sequence"],
                    label=current_object["name"],
                    num_sensations_to_converge=current_object[
                        "numSensationsToSingleConverge"],
                    ground_truth=current_object["ground_truth_image"])

            elif DECODER_ARCHITECTURE == "abstract_SDR_decoder":

                contains_empty_predictions = predict_GridCellNet_broad_patch_reps(
                    net, prediction_sequence=current_object["prediction_sequence"],
                    touch_sequence=current_object["touch_sequence"],
                    label=current_object["name"],
                    num_sensations_to_converge=current_object[
                        "numSensationsToSingleConverge"],
                    ground_truth=current_object["ground_truth_image"])

            objects_with_convergence += 1
            objects_with_empty_predictions += contains_empty_predictions

    print("Of " + str(objects_with_convergence) + " valid objects "
          + str(objects_with_empty_predictions) + " have empty predictions. Percent: "
          + str(objects_with_empty_predictions / objects_with_convergence))
