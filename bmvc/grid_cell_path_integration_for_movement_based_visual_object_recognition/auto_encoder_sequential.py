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
Used to generate sparse-feature vectors based on an
auto-encoders middle layers
Can also be used to train a decoder that can generate
image reconstructions based on these SDRs
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from nupic.torch.modules import KWinners2d, update_boost_strength

torch.manual_seed(1)
np.random.seed(1)

train_net_bool = True
ARCHITECTURE = "MLPAutoEncoder"  # Options are "MLPAutoEncoder" or "PatchDecoder"
RANDOM_BOOL = False  # Set to true if the pixel-patches should be shuffled; used
# to determine the degree to which a system relies on spatially insensitive processing
NUM_IMAGE_EXAMPLES = 2000  # Number of training and evaluation-dataset images for the
# patchwise auto-encoder to create; these are used in downstream tasks such as for the
# SDR-based classifiers; recommend 2k

data_set = "mnist"
generate_sdr_patches_bool = True
CROSS_VAL_SPLIT = 0.1  # Split used for the training data-set
num_epochs = 10
batch_size = 64
percent_on = 0.15
boost_strength = 20.0


class MLPAutoEncoder(torch.nn.Module):
    """
    Auto encoder that learns to re-construct a patch of pixels;
    Mid-layer representations are constrained to be sparse with
    k-WTA, and later binarized for use as an input to downstream classifiers
    NB they are not binarized at this point in order to preserve gradients
    """
    def __init__(self):
        super(MLPAutoEncoder, self).__init__()
        self.dense1_encode = nn.Linear(in_features=7 * 7, out_features=25)
        self.dense2_encode = nn.Linear(in_features=25, out_features=128)
        self.k_winner = KWinners2d(channels=128, percent_on=percent_on,
                                   boost_strength=boost_strength, local=True)
        self.dense1_decode = nn.Linear(in_features=128, out_features=7 * 7)

    def encode(self, x):
        x = x.reshape(-1, 7 * 7)
        x = F.relu(self.dense1_encode(x))
        x = F.relu(self.dense2_encode(x))

        x = x.reshape(-1, 128, 1, 1)
        x = self.k_winner(x)
        x = x.reshape(-1, 128)

        return x

    def decode(self, x):

        x = torch.sigmoid(self.dense1_decode(x))
        x = x.view(-1, 7, 7)

        return x

    def forward(self, x):

        x = self.encode(x)
        x = self.decode(x)

        return x

    def extract_sdrs(self, x):

        x = self.encode(x)
        indices = x > 0
        x[indices] = 1

        return x


class PatchDecoder(torch.nn.Module):
    """
    Decoder that learns to take *binary* sparse representations and
    reconstruct a pixel patch; for use in visualising the predictions of
    GridCellNet
    """
    def __init__(self):
        super(PatchDecoder, self).__init__()
        self.dense1_decode = nn.Linear(in_features=128, out_features=128)
        self.dense2_decode = nn.Linear(in_features=128, out_features=7 * 7)

    def forward(self, x):
        x = self.dense1_decode(x)
        x = torch.sigmoid(self.dense2_decode(x))
        x = x.view(-1, 7, 7)

        return x


def sample_patches(input_images):
    """
    Take patches from available images and turn them into a long
    list for use by the auto-encoder (e.g. 16 patches per image)
    """

    assembled_patches = []

    # Iterate over images
    for image_iter in range(len(input_images)):

        # Iterate over patches in the image
        for patch_width_iter in range(4):

            for patch_height_iter in range(4):

                assembled_patches.append(input_images.detach().numpy()[image_iter][
                    patch_width_iter * 7:(patch_width_iter + 1) * 7,
                    patch_height_iter * 7:(patch_height_iter + 1) * 7])

    return assembled_patches


def initialize(sample_patches_bool):

    # Data not normalized to ensure no issues with pixel-level reconstruction
    normalize = None

    # "Sources" are the original image that needs to be reconstructed
    base_net_training_sources = datasets.MNIST(
        "data", train=True, download=True,
        transform=normalize).train_data.float() / 255
    base_net_training_labels = datasets.MNIST(
        "data", train=True, download=True,
        transform=normalize).train_labels

    traing_len = len(base_net_training_sources)

    indices = range(traing_len)
    val_split = int(np.floor(CROSS_VAL_SPLIT * traing_len))
    base_net_train_idx, crossval_train_idx = indices[val_split:], indices[:val_split]

    base_net_training_sources = base_net_training_sources[base_net_train_idx]
    base_net_training_labels = base_net_training_labels[base_net_train_idx]

    sdr_classifiers_training_sources = datasets.MNIST(
        "data", train=True,
        download=True,
        transform=normalize).train_data.float()[crossval_train_idx] / 255
    sdr_classifiers_training_labels = datasets.MNIST(
        "data", train=True,
        download=True, transform=normalize).train_labels[crossval_train_idx]

    sdr_classifiers_testing_sources = datasets.MNIST(
        "data", train=False,
        download=True, transform=normalize).test_data.float() / 255
    sdr_classifiers_testing_labels = datasets.MNIST(
        "data", train=False,
        download=True, transform=normalize).test_labels

    # Check data-set sizes; if these fail, something has gone wrong
    assert len(base_net_training_sources) == len(base_net_training_labels), \
        "Need equal number of images and labels"
    assert (len(sdr_classifiers_training_sources)
            == len(sdr_classifiers_training_labels)), \
        "Need equal number of images and labels"
    assert (len(sdr_classifiers_testing_sources)
            == len(sdr_classifiers_testing_labels)), \
        "Need equal number of images and labels"

    # Sample-patches is used for training an auto-encoder/decoder, but not if using
    # the generate patchwise-SDR function
    if sample_patches_bool is True:
        base_net_training_sources = torch.FloatTensor(
            sample_patches(base_net_training_sources))
        sdr_classifiers_training_sources = torch.FloatTensor(
            sample_patches(sdr_classifiers_training_sources))
        sdr_classifiers_testing_sources = torch.FloatTensor(
            sample_patches(sdr_classifiers_testing_sources))

    # If training a decoder, load SDRs that will be used to reconstruct pixel patches;
    # Note the base-net/SDR classifier training split is always used, and that this
    # cannot be called unless an auto-encoder has already been trained and evaluated
    # to develop the necessary SDRs
    if ARCHITECTURE == "PatchDecoder":
        base_net_training_sdr_sources = np.load(
            "python2_htm_docker/docker_dir/training_and_testing_data/"
            + data_set + "_sdrs_base_net_training.npy")
        sdr_classifiers_training_sdr_sources = np.load(
            "python2_htm_docker/docker_dir/training_and_testing_data/"
            + data_set + "_sdrs_sdr_classifiers_training.npy")

        # Reshape the SDR data to be aligned with the pixel-patches
        base_net_training_sdr_sources = np.moveaxis(
            base_net_training_sdr_sources, 1, -1)
        sdr_classifiers_training_sdr_sources = np.moveaxis(
            sdr_classifiers_training_sdr_sources, 1, -1)

        base_net_training_sdr_sources = np.reshape(
            base_net_training_sdr_sources,
            (np.shape(base_net_training_sdr_sources)[0] * 4 * 4,
                np.shape(base_net_training_sdr_sources)[-1]))
        sdr_classifiers_training_sdr_sources = np.reshape(
            sdr_classifiers_training_sdr_sources,
            (np.shape(sdr_classifiers_training_sdr_sources)[0] * 4 * 4,
                np.shape(sdr_classifiers_training_sdr_sources)[-1]))

    else:
        base_net_training_sdr_sources = None
        sdr_classifiers_training_sdr_sources = None

    assert len(base_net_training_sources) != len(sdr_classifiers_training_sources), \
        "Ensure the data-sets are not duplicates"
    assert len(base_net_training_sources) != len(sdr_classifiers_testing_sources), \
        "Ensure the data-sets are not duplicates"
    assert (len(sdr_classifiers_testing_sources)
            != len(sdr_classifiers_training_sources)), \
        "Ensure the data-sets are not duplicates"

    return (base_net_training_sources, sdr_classifiers_training_sources,
            sdr_classifiers_testing_sources, base_net_training_labels,
            sdr_classifiers_training_labels, sdr_classifiers_testing_labels,
            base_net_training_sdr_sources, sdr_classifiers_training_sdr_sources)


def train_decoder_net(net, training_sources, training_sdr_sources):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    training_sources = training_sources[:np.shape(training_sdr_sources)[0], :, :]

    net.train()

    print("\nTraining a decoder network given SDR inputs")

    for epoch in range(num_epochs):

        shuffle_indices = torch.randperm(np.shape(training_sdr_sources)[0])
        training_sources = training_sources[shuffle_indices, :]
        training_sdr_sources = training_sdr_sources[shuffle_indices, :]

        running_loss = 0

        for batch_iter in range(math.ceil(len(training_sdr_sources) / batch_size)):

            pixel_patch_sources = training_sources[
                batch_iter * batch_size:min(
                    (batch_iter + 1) * batch_size, len(training_sdr_sources))]
            sdr_patch_sources = torch.FloatTensor(
                training_sdr_sources[batch_iter * batch_size:min(
                    (batch_iter + 1) * batch_size, len(training_sdr_sources))])

            optimizer.zero_grad()

            reconstructed = net(sdr_patch_sources)

            loss = criterion(reconstructed, pixel_patch_sources)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("\nEpoch:" + str(epoch))
        print("Training loss is " + str(running_loss / len(training_sources)))

    print("Saving network state...")
    torch.save(net.state_dict(), "saved_networks/" + data_set + "_patch_decoder.pt")

    print("Finished Training")

    return None


def train_autoencoder_net(net, training_sources):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.train()

    print("\nTraining an auto-encoder network for later generation of SDRs")

    for epoch in range(num_epochs):

        running_loss = 0

        for batch_iter in range(math.ceil(len(training_sources) / batch_size)):

            batch_sources = training_sources[
                batch_iter
                * batch_size:min((batch_iter + 1) * batch_size, len(training_sources))]

            optimizer.zero_grad()

            reconstructed = net(batch_sources)

            loss = criterion(reconstructed, batch_sources)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        net.apply(update_boost_strength)

        duty_cycle = (net.k_winner.duty_cycle).numpy()
        duty_cycle = (net.k_winner.duty_cycle[0, :, 0, 0]).numpy()
        print("\nEpoch:" + str(epoch))
        print("Training loss is " + str(running_loss / len(training_sources)))
        print("Mean duty cycle : " + str(np.mean(duty_cycle)))
        print("Stdev duty cycle: " + str(np.std(duty_cycle)))

    print("Saving network state...")
    torch.save(net.state_dict(), "saved_networks/"
               + data_set + "_patch_autoencoder.pt")

    print("Finished Training")

    return None


def evaluate_auto_encoder(net, source):

    net.load_state_dict(torch.load("saved_networks/"
                        + data_set + "_patch_autoencoder.pt"))

    net.eval()

    print("Outputing a sample of the trained auto-encodrs predictions")
    num_example_patches = 10

    reconstructed = net(source[0:num_example_patches])

    for image_iter in range(num_example_patches):

        plt.imsave("output_images/" + str(image_iter) + "_original_patch.png",
                   source.detach().numpy()[image_iter])
        plt.imsave("output_images/" + str(image_iter) + "_reconstructed_patch.png",
                   reconstructed.detach().numpy()[image_iter])

    return None


def generate_patch_wise_sdrs(net, input_images, all_labels, data_split,
                             randomize_order_bool=False):
    """
    Use a trained auto-encoder to generate binary SDRs corresponding to pixel-patches
    """

    net.load_state_dict(torch.load("saved_networks/" + data_set
                                   + "_patch_autoencoder.pt"))

    net.eval()

    input_images_copy = torch.clone(input_images)

    print("\nGenerating SDRs for " + str(NUM_IMAGE_EXAMPLES)
          + " images on the " + data_split + " data split.")

    all_image_sdrs = []
    all_labels_output = []

    if os.path.exists("output_images/" + data_split + "/") is False:
        try:
            os.mkdir("output_images/" + data_split + "/")
        except OSError:
            pass

    for image_iter in range(NUM_IMAGE_EXAMPLES):

        image_sdrs = np.zeros((128, 4, 4))
        image_reconstruct = np.zeros((28, 28))

        # Values used if shuffling the arrangement of features/patches
        # Flips the order of the features to guarentee breaking any
        # local arrangements of features
        random_width_locations = list(range(4))
        random_width_locations = np.flip(random_width_locations)
        random_height_locations = list(range(4))
        random_height_locations = np.flip(random_height_locations)

        for patch_width_iter in range(4):

            for patch_height_iter in range(4):

                input_patch = input_images[image_iter][
                    patch_width_iter * 7:(patch_width_iter + 1) * 7,
                    patch_height_iter * 7:(patch_height_iter + 1) * 7]

                input_patch = input_patch[None, :, :]

                if randomize_order_bool:

                    # If true, takes the SDR from a given pixel patch, but places it at
                    # a different location w.r.t. the original true image
                    patch_width_index = random_width_locations[patch_width_iter]
                    patch_height_index = random_height_locations[patch_height_iter]

                else:
                    patch_width_index = patch_width_iter
                    patch_height_index = patch_height_iter

                # Visualise the reconstructed image, including with any potential
                # randomization of the patch arrangements
                image_reconstruct[
                    patch_width_index * 7:(patch_width_index + 1) * 7,
                    patch_height_index * 7:(patch_height_index + 1) * 7] = \
                    net(input_patch).detach().numpy()

                image_sdrs[:, patch_width_index, patch_height_index] = \
                    net.extract_sdrs(input_patch).detach().numpy()

        current_label = all_labels[image_iter].detach().numpy()
        all_labels_output.append(current_label)
        all_image_sdrs.append(image_sdrs)

        # Visualize a handful of the generated images
        if image_iter <= 10:

            plt.imsave("output_images/" + data_split + "/" + str(image_iter)
                       + "_original_image_label_" + str(current_label) + ".png",
                       input_images_copy.detach().numpy()[image_iter])
            plt.imsave("output_images/" + data_split + "/" + str(image_iter)
                       + "_reconstructed_image_label_" + str(current_label) + ".png",
                       image_reconstruct)

    return all_image_sdrs, all_labels_output, input_images.detach().numpy()


def make_general_dir():

    if os.path.exists("output_images/") is False:
        try:
            os.mkdir("output_images/")
        except OSError:
            pass

    if os.path.exists("saved_networks/") is False:
        try:
            os.mkdir("saved_networks/")
        except OSError:
            pass


def make_generate_sdr_dir():

    if os.path.exists(
            "python2_htm_docker/docker_dir/training_and_testing_data/") is False:
        try:
            os.mkdir("python2_htm_docker/docker_dir/training_and_testing_data/")
        except OSError:
            pass


if __name__ == "__main__":

    make_general_dir()

    (base_net_training_sources, sdr_classifiers_training_sources,
        sdr_classifiers_testing_sources, base_net_training_labels,
        sdr_classifiers_training_labels, sdr_classifiers_testing_labels,
        base_net_training_sdr_sources,
        sdr_classifiers_training_sdr_sources) = initialize(sample_patches_bool=True)

    # NB the patch-decoder is only ever "evaluated"/used by
    # visualise_GridCellNet_predictions.py, hence why there is no other option here
    if ARCHITECTURE == "PatchDecoder":

        net = PatchDecoder()

        train_decoder_net(net, base_net_training_sources,
                          base_net_training_sdr_sources)

    else:

        net = MLPAutoEncoder()

        if train_net_bool is True:
            print("Training new network")
            train_autoencoder_net(net, base_net_training_sources)
            print("Evaluating newly trained network")
            evaluate_auto_encoder(net, source=sdr_classifiers_training_sources)

        elif train_net_bool is False:
            print("Evaluating previously trained network")
            evaluate_auto_encoder(net, source=sdr_classifiers_training_sources)

        if generate_sdr_patches_bool is True:

            make_generate_sdr_dir()

            (base_net_training_sources, sdr_classifiers_training_sources,
                sdr_classifiers_testing_sources, base_net_training_labels,
                sdr_classifiers_training_labels, sdr_classifiers_testing_labels,
                _, _) = initialize(sample_patches_bool=False)

            print("Generating patch-based SDRs for base-net training data")
            data_split = "base_net_training"
            all_image_sdrs, all_labels_output, input_images = generate_patch_wise_sdrs(
                net, input_images=base_net_training_sources,
                all_labels=base_net_training_labels, data_split=data_split,
                randomize_order_bool=False)

            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_SDRs_" + data_split, all_image_sdrs)
            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_labels_" + data_split, all_labels_output)
            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_images_" + data_split, input_images)

            print("Generating patch-based SDRs for SDR-classifier *training* data")
            data_split = "SDR_classifiers_training"
            all_image_sdrs, all_labels_output, input_images = generate_patch_wise_sdrs(
                net, input_images=sdr_classifiers_training_sources,
                all_labels=sdr_classifiers_training_labels,
                data_split=data_split, randomize_order_bool=False)

            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_SDRs_" + data_split, all_image_sdrs)
            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_labels_" + data_split, all_labels_output)
            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_images_" + data_split, input_images)

            print("Generating patch-based SDRs for SDR-classifier *testing* data")
            data_split = "SDR_classifiers_testing"
            all_image_sdrs, all_labels_output, input_images = generate_patch_wise_sdrs(
                net, input_images=sdr_classifiers_testing_sources,
                all_labels=sdr_classifiers_testing_labels, data_split=data_split,
                randomize_order_bool=RANDOM_BOOL)

            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_SDRs_" + data_split, all_image_sdrs)
            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_labels_" + data_split, all_labels_output)
            np.save("python2_htm_docker/docker_dir/training_and_testing_data/"
                    + data_set + "_images_" + data_split, input_images)
