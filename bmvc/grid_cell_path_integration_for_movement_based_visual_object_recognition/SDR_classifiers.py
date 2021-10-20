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
Trains basic classifiers on sparse feature vectors extracted derived from images
"""

import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

seed_val = 1
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)

DATASET = "mnist"  # Options are "mnist"
SAMPLES_PER_CLASS_LIST = [1, 5, 10, 20]  # How many samples per class to use in
# training the classifier; can be provided as a single-value list (e.g. [5]), or
# multi-valued list, in which case a classifier will be trained for each number of
# sample classes
CLASSIFIER_TYPE = "rnn"  # Options are "rnn" or "knn"
INPUT_GRID_DIMENSION = 5  # Options are 4 or 5, depending on the kind of pre-processing
# of the input used, i.e. patches of 7x7 pixels (dim=4), or features extracted from a
# pre-trained CNN's middle layers (dim=5)
MOVEMENT_INFO_BOOL = True  # Concatenates the movement or location information
# of the sensor to the features themselves, and provides this expanded feature
# to the classifier
MOVEMENT_INFO_TYPE = "vector_displacement"  # Options are "x_y_displacement",
# "vector_displacement", and "absolute_location"; should be set to None if
# MOVEMENT_INFO_BOOL is False
ARBITRARY_SDR_ORDER_BOOL = True  # Option to shuffle the order of the SDRs for each
# example; this is used to determine the robustness of a classifier to a stream of
# inputs of arbitrary order
USE_RASTER_BOOL = False  # Follow a fixed rastor sequence if True and
# ARBITRARY_SDR_ORDER_BOOL set to False
BLOCKED_TRAINING_EXAMPLES_BOOL = False  # If True, block training examples together
# (e.g. 50 epochs of training on classes 0 through 4, followed by 50 epochs on classes
# 5 through 9); used to evaluate robustness to continual learning; NB
# samples_per_class, learning rate etc. are already set by default, rather than the
# parameters here
FALSE_MOVEMENT_INFO_AT_EVALUATION = False  # NB this is only supported with vector
# movement type information. Provide movement information correctly during learning,
# but use a plausible (albeit incorrect) touch sequence to derive movement information
# at inference; helps demonstrate to what extent the RNN is actually using
# self-movement to inform representations
FIXED_STARTING_POSITION_BOOL = False  # When the order of features is arbitrary, always
# begin at the same location in the external reference frame of the image during
# training and evaluation; only relevant if ARBITRARY_SDR_ORDER_BOOL is True
if FIXED_STARTING_POSITION_BOOL is True:
    STARTING_POSITION = np.random.randint(0, INPUT_GRID_DIMENSION
                                          * INPUT_GRID_DIMENSION)

# Hyperparameters for neural networks
EPOCHS = 50
WEIGHT_DECAY = 0.001  # Recommend 0.001 for RNN
BATCH_SIZE = 128  # Recommend 128
LR_LIST = [0.005, 0.01]  # Learning rates to try for the neural network; can specify
# just a single-value list if desired

# Hyperparameters for k-NN
KNN_PROGRESSIVE_SENSATIONS_BOOL = False  # Whether to evaluate the k-NN classifier
# where progressively more input points are given (from just 1 up to the maximum of
# INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION)
# This provides an indication of how much additional sensations improves performance
N_NEIGHBOURS_LIST = [1]  # Number of neighbours to try for the k-NN, e.g.
# list(range(1, 11))
TEST_EXAMPLES_PER_CLASS = 100  # Typically 100


class RNNModel(torch.nn.Module):
    """
    Basic RNN to test the learning of sequential inputs.
    """
    def __init__(self):
        super(RNNModel, self).__init__()

        self.hidden_size = 128
        self.num_layers = 1
        self.rnn = torch.nn.LSTM(input_size=128 + int(MOVEMENT_INFO_BOOL)
                                 + int(MOVEMENT_INFO_TYPE == "vector_displacement"
                                       or MOVEMENT_INFO_TYPE == "x_y_displacement"),
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, 10)

    def forward(self, x):

        hidden_t = torch.zeros(self.num_layers, np.shape(x)[0], self.hidden_size)
        cell_t = torch.zeros(self.num_layers, np.shape(x)[0], self.hidden_size)

        x = x.reshape(-1, 128 + int(MOVEMENT_INFO_BOOL)
                      + int(MOVEMENT_INFO_TYPE == "vector_displacement"
                            or MOVEMENT_INFO_TYPE == "x_y_displacement"),
                      INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION)
        x = torch.FloatTensor(np.moveaxis(x.numpy(), 2, 1))  # Swap feature and
        # sequence axis

        out, (hidden_t, cell_t) = self.rnn(x, (hidden_t, cell_t))
        out = out[:, -1, :]  # Take the final representation
        out = self.fc(out)

        return out


def calculate_discrete_displacement(previous_location, new_location):
    """
    Determine the x and y displacement (measured as discrete movements in a
    INPUT_GRID_DIMENSIONxINPUT_GRID_DIMENSION grid) between the previous
    and new location of the sensor
    """

    # x is treated as indexing columns, y as the rows of the
    # INPUT_GRID_DIMENSIONxINPUT_GRID_DIMENSION grid
    # Derive from the absolute location (indexed 0 to 24)
    x_prev = previous_location % INPUT_GRID_DIMENSION
    y_prev = math.floor(previous_location / INPUT_GRID_DIMENSION)

    # TODO refactor this so I don't repeat this calculation in code
    x_new = new_location % INPUT_GRID_DIMENSION
    y_new = math.floor(new_location / INPUT_GRID_DIMENSION)

    x_diff = x_new - x_prev  # Moving right along the
    # INPUT_GRID_DIMENSIONxINPUT_GRID_DIMENSION grid is considered positive
    y_diff = y_prev - y_new  # Moving up along the
    # INPUT_GRID_DIMENSIONxINPUT_GRID_DIMENSION grid is considered positive

    return y_diff, x_diff


def calculate_vector_displacement(previous_location, new_location):

    y_diff, x_diff = calculate_discrete_displacement(previous_location, new_location)

    euclid = math.sqrt((x_diff ** 2) + (y_diff ** 2))

    # Theta is determined from the right-hand x-axis using the inverse cosine
    # If the movement goes downward, theta is given as a negative value
    # Thus the network can learn that theta near 0 corresponds to moving right,
    # near pi/-pi to moving left, and positive and negative to moving up and
    # down respecitively
    if y_diff < 0:
        theta_flip = -1.0
    else:
        theta_flip = 1.0

    theta = theta_flip * math.acos(x_diff / euclid)

    return euclid, theta


def sub_sample_classes(input_data, labels, samples_per_class, sanity_check=None):
    """
    As we're evaluating few-shot learning, take a sub-sample while ensuring an equal
    number of each class
    """
    input_data_samples = []
    label_samples = []

    print("Loading " + str(samples_per_class) + " examples per class")

    if sanity_check == "one_class_training":
        print("\nAs a sanity check, loading data for only a single class")
        num_classes = 1
    else:
        num_classes = 10

    for class_iter in range(num_classes):
        indices = np.nonzero(labels == class_iter)

        input_data_samples.extend(input_data[indices][0:samples_per_class])
        label_samples.extend(labels[indices][0:samples_per_class])

        assert len(labels[indices][0:samples_per_class]) == samples_per_class, \
            "Insufficient loaded samples"

    return input_data_samples, label_samples


def shuffle_sdr_order(input_data_samples, random_indices, eval_data_bool):
    """
    Shuffles the order of the input SDRs (total of
    INPUT_GRID_DIMENSION x INPUT_GRID_DIMENSION)
    """
    sdr_shuffled_input_data_samples = []

    feature_dim = (128 + int(MOVEMENT_INFO_BOOL)
                   + int(MOVEMENT_INFO_TYPE == "vector_displacement"
                         or MOVEMENT_INFO_TYPE == "x_y_displacement"))

    for image_iter in range(len(input_data_samples)):

        temp_random_indices = np.copy(random_indices)
        false_random_indices = np.copy(random_indices)  # Used if providing
        # false movement information at inference time to the classifier
        if FALSE_MOVEMENT_INFO_AT_EVALUATION:
            np.random.shuffle(false_random_indices)  # Always shuffled for each image

        if ARBITRARY_SDR_ORDER_BOOL is True:

            np.random.shuffle(temp_random_indices)  # Re-shuffle the SDRs for each
            # image; otherwise the same fixed sequence is used to re-order them

            if FIXED_STARTING_POSITION_BOOL is True:

                # Remove the STARTING_POSITION occurence from the random sequence
                # and place it at the beginning
                temp_random_indices = temp_random_indices[temp_random_indices
                                                          != STARTING_POSITION]
                temp_random_indices = np.insert(temp_random_indices,
                                                0, [STARTING_POSITION])

        temp_sdr_array = np.reshape(input_data_samples[image_iter],
                                    (128, INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION))

        # Include movement/location information as a part of the feature itself
        if MOVEMENT_INFO_BOOL is True:

            if MOVEMENT_INFO_TYPE == "absolute_location":
                # Provide absolute location information

                temp_sdr_array = np.concatenate(
                    (temp_sdr_array,
                     np.transpose(temp_random_indices[:, None])), axis=0)

            else:

                # First sensation has no movement
                displacements_array = [[0, 0]]

                if MOVEMENT_INFO_TYPE == "x_y_displacement":

                    for touch_iter in range(len(temp_random_indices) - 1):
                        y_diff_temp, x_diff_temp = calculate_discrete_displacement(
                            temp_random_indices[touch_iter],
                            temp_random_indices[touch_iter + 1])

                        displacements_array.append([y_diff_temp, x_diff_temp])

                elif MOVEMENT_INFO_TYPE == "vector_displacement":

                    for touch_iter in range(len(random_indices) - 1):

                        # Note that false movement information would only ever be
                        # provided at inference
                        if FALSE_MOVEMENT_INFO_AT_EVALUATION and eval_data_bool:
                            euclid, theta = calculate_vector_displacement(
                                false_random_indices[touch_iter],
                                false_random_indices[touch_iter + 1])
                        else:
                            euclid, theta = calculate_vector_displacement(
                                temp_random_indices[touch_iter],
                                temp_random_indices[touch_iter + 1])

                        displacements_array.append([euclid, theta])

                temp_sdr_array = np.concatenate(
                    (temp_sdr_array, np.transpose(displacements_array)), axis=0)

        random_sdr_array = temp_sdr_array[:, temp_random_indices]
        sdr_shuffled_input_data_samples.append(
            np.reshape(random_sdr_array,
                       (feature_dim * INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION)))

    return sdr_shuffled_input_data_samples


def truncate_sdr_samples(input_data_samples, truncation_point):
    """
    Truncate the input SDRs, so as to evaluate e.g. how well a k-NN performs when given
    only 3 out of the total INPUT_GRID_DIMENSION*INPUT_GRID_DIMENSION input features
    """
    truncated_input_data_samples = []

    for image_iter in range(len(input_data_samples)):

        temp_sdr_array = np.reshape(
            input_data_samples[image_iter],
            (128, INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION))
        truncated_sdr_array = temp_sdr_array[:, 0:truncation_point + 1]
        truncated_input_data_samples.append(
            np.reshape(
                truncated_sdr_array,
                ((128) * (truncation_point + 1))))

    return truncated_input_data_samples


def load_data(data_section, random_indices, samples_per_class=5, sanity_check=None,
              dataset=DATASET, eval_data_bool=False):

    input_data = np.load("python2_htm_docker/docker_dir/training_and_testing_data/"
                         + DATASET + "_SDRs_" + data_section + ".npy")
    labels = np.load("python2_htm_docker/docker_dir/training_and_testing_data/"
                     + DATASET + "_labels_" + data_section + ".npy")

    print("\nLoading data from " + data_section)

    input_data_samples, label_samples = sub_sample_classes(input_data, labels,
                                                           samples_per_class,
                                                           sanity_check=None)

    input_data_samples = shuffle_sdr_order(input_data_samples,
                                           random_indices, eval_data_bool)  # Note
    # this still maintains the order of the examples, just not their features

    return input_data_samples, label_samples


def knn(n_neighbors, training_data, training_labels, testing_data, testing_labels):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(training_data, training_labels)
    acc = knn.score(testing_data, testing_labels)

    print("Accuracy of k-NN classifier " + str(acc))

    return acc


def knn_progressive_senations(n_neighbors, training_data, training_labels,
                              testing_data, testing_labels):

    acc_list = []

    print("\nTruncating the number of sensations/SDR locations provided")

    for truncation_iter in range(INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION):

        truncated_training_data = truncate_sdr_samples(training_data, truncation_iter)
        truncated_testing_data = truncate_sdr_samples(testing_data, truncation_iter)

        acc_list.append(knn(n_neighbors, truncated_training_data, training_labels,
                            truncated_testing_data, testing_labels))

    print("All accuracies across truncation levels")
    print(acc_list)

    plt.scatter(list(range(1, 26)), acc_list)
    plt.ylim(0, 1)
    plt.show()

    return None


def train(net, training_data, training_labels, optimizer, criterion, epoch):
    net.train()

    shuffle_indices = torch.randperm(len(training_labels))
    training_data = training_data[shuffle_indices, :]
    training_labels = training_labels[shuffle_indices]

    for batch_i in range(math.ceil(len(training_labels) / BATCH_SIZE)):

        training_batch_data = (
            training_data[
                batch_i
                * BATCH_SIZE:min((batch_i + 1) * BATCH_SIZE, len(training_labels))])
        training_batch_labels = (
            training_labels[
                batch_i
                * BATCH_SIZE:min((batch_i + 1) * BATCH_SIZE, len(training_labels))])

        optimizer.zero_grad()

        outputs = net(training_batch_data)
        loss = criterion(outputs, training_batch_labels)
        loss.backward()

        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)

        optimizer.step()

    # Accuracy on all data
    training_acc, total_loss = evaluate(net, training_data, training_labels, criterion)

    return training_acc, total_loss


def evaluate(net, testing_data, testing_labels, criterion):
    net.eval()

    with torch.no_grad():
        outputs = net(testing_data)

        testing_matches = torch.sum(torch.argmax(outputs,
                                    dim=1) == testing_labels)
        loss = criterion(outputs, testing_labels)

        testing_acc = 100 * (testing_matches).item() / len(testing_labels)

    return testing_acc, loss.item()


def train_net(net, training_data, training_labels,
              testing_data, testing_labels, lr, samples_per_class):

    (training_data, training_labels,
        testing_data, testing_labels) = (torch.FloatTensor(training_data),
                                         torch.LongTensor(training_labels),
                                         torch.FloatTensor(testing_data),
                                         torch.LongTensor(testing_labels))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                 weight_decay=WEIGHT_DECAY)

    training_accuracies = []
    testing_accuracies = []

    training_losses = []
    testing_losses = []

    for epoch in range(EPOCHS):

        training_acc, training_loss = train(
            net, training_data, training_labels, optimizer, criterion, epoch)
        testing_acc, testing_loss = evaluate(
            net, testing_data, testing_labels, criterion)

        print("\nEpoch:" + str(epoch))
        print("Training accuracy is " + str(training_acc))
        print("Training loss is " + str(training_loss))
        print("Testing accuracy is " + str(testing_acc))
        print("Testing loss is " + str(testing_loss))

        training_accuracies.append(training_acc)
        testing_accuracies.append(testing_acc)

        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

    if os.path.exists("results/" + str(samples_per_class)
                      + "_samples_plots") is False:
        try:
            os.mkdir("results/" + str(samples_per_class) + "_samples_plots")
        except OSError:
            pass

    plt.plot(training_accuracies, label="Training")
    plt.plot(testing_accuracies, label="Testing")
    plt.legend()
    plt.ylim(0, 100)
    plt.savefig("results/" + str(samples_per_class) + "_samples_plots/"
                + str(lr) + "_lr_accuracies.png")
    plt.clf()

    plt.plot(training_losses, label="Training")
    plt.plot(testing_losses, label="Testing")
    plt.legend()
    plt.savefig("results/" + str(samples_per_class) + "_samples_plots/"
                + str(lr) + "_lr_losses.png")
    plt.clf()

    print("Finished Training")
    return testing_acc, testing_accuracies


def generic_setup(samples_per_class):
    # Note the same fixed, random sampling of the input is used across all examples
    # in both training and testing, unless ARBITRARY_SDR_ORDER_BOOL==True
    random_indices = np.arange(INPUT_GRID_DIMENSION * INPUT_GRID_DIMENSION)

    if USE_RASTER_BOOL is False:
        np.random.shuffle(random_indices)

    training_data, training_labels = load_data(data_section="SDR_classifiers_training",
                                               random_indices=random_indices,
                                               samples_per_class=samples_per_class,
                                               sanity_check=None, eval_data_bool=False)

    # Note unless specified otherwise, the full test-dataset is not used for
    # evaluation, as this would take too long for GridCellNet
    testing_data, testing_labels = load_data(data_section="SDR_classifiers_testing",
                                             random_indices=random_indices,
                                             samples_per_class=TEST_EXAMPLES_PER_CLASS,
                                             sanity_check=None, eval_data_bool=True)

    return training_data, training_labels, testing_data, testing_labels


def learn_blocked_data(net, training_data, training_labels, testing_data,
                       testing_labels, lr, samples_per_class,
                       training_data_bottom_class, training_data_top_class):

    print("\nTraining a classifier using classes " + str(training_data_bottom_class)
          + " through " + str(training_data_top_class - 1))

    training_data_class_section = training_data[training_data_bottom_class
                                                * samples_per_class
                                                : training_data_top_class
                                                * samples_per_class]
    training_labels_class_section = training_labels[training_data_bottom_class
                                                    * samples_per_class
                                                    : training_data_top_class
                                                    * samples_per_class]

    # NB we evaluate on the entire test data-set across all classes
    _, testing_accuracies = train_net(net, training_data_class_section,
                                      training_labels_class_section,
                                      testing_data, testing_labels,
                                      lr, samples_per_class)

    return testing_accuracies


def blocked_training(training_data, training_labels, testing_data,
                     testing_labels, samples_per_class):

    # NB that the sub_sample_classes method already, by default, blocks class examples
    # together, which is undone by shuffling/interleaving example order on each epoch
    # in typical training

    net = RNNModel()

    assert len(LR_LIST) == 1, "Only provide one LR-value for the conintual" \
        "learning experimental set-up"
    lr = LR_LIST[0]  # Assumes one value was given in the list

    # NB the training_data_top_class range is exclusive, while the bottom_class index
    # is inclusive

    # ==== 5-5 block ====
    testing_acc_across_all_classes = []

    # 1st block
    testing_accuracies = learn_blocked_data(net, training_data, training_labels,
                                            testing_data, testing_labels, lr,
                                            samples_per_class,
                                            training_data_bottom_class=0,
                                            training_data_top_class=5)

    testing_acc_across_all_classes.extend(testing_accuracies)
    print("First block results:")
    print(testing_accuracies)

    # 2nd block
    testing_accuracies = learn_blocked_data(net, training_data, training_labels,
                                            testing_data, testing_labels, lr,
                                            samples_per_class,
                                            training_data_bottom_class=5,
                                            training_data_top_class=10)

    testing_acc_across_all_classes.extend(testing_accuracies)
    print("Second block results:")
    print(testing_accuracies)

    print("Combined results:")
    print(testing_acc_across_all_classes)

    if os.path.exists("results/5_5_blocked_training/") is False:
        try:
            os.mkdir("results/5_5_blocked_training/")
        except OSError:
            pass

    plt.plot(testing_acc_across_all_classes)
    plt.savefig("results/5_5_blocked_training/accuracy.png")
    plt.clf()

    # ==== 9-1 block ====

    # Re-initialize network
    net = RNNModel()

    testing_acc_across_all_classes = []

    # 1st block
    testing_accuracies = learn_blocked_data(net, training_data, training_labels,
                                            testing_data, testing_labels, lr,
                                            samples_per_class,
                                            training_data_bottom_class=0,
                                            training_data_top_class=9)

    testing_acc_across_all_classes.extend(testing_accuracies)
    print("First block results:")
    print(testing_accuracies)

    # 2nd block
    testing_accuracies = learn_blocked_data(net, training_data, training_labels,
                                            testing_data, testing_labels, lr,
                                            samples_per_class,
                                            training_data_bottom_class=9,
                                            training_data_top_class=10)

    testing_acc_across_all_classes.extend(testing_accuracies)
    print("Second block results:")
    print(testing_accuracies)

    print("Combined results:")
    print(testing_acc_across_all_classes)

    if os.path.exists("results/9_1_blocked_training/") is False:
        try:
            os.mkdir("results/9_1_blocked_training/")
        except OSError:
            pass

    plt.plot(testing_acc_across_all_classes)
    plt.savefig("results/9_1_blocked_training/accuracy.png")
    plt.clf()

    return None


def run_over_hyperparameters(training_data, training_labels,
                             testing_data, testing_labels):

    acc_dic = {}

    if CLASSIFIER_TYPE == "knn":

        if KNN_PROGRESSIVE_SENSATIONS_BOOL is True:

            n_neighbors = N_NEIGHBOURS_LIST[0]
            print("Performing k-NN classification with a progressive number of "
                  "sensations and  # neighbours = " + str(n_neighbors))
            knn_progressive_senations(n_neighbors, training_data, training_labels,
                                      testing_data, testing_labels)

        else:

            for n_neighbors in N_NEIGHBOURS_LIST:

                acc_dic["n_neighbors_" + str(n_neighbors)] = knn(n_neighbors,
                                                                 training_data,
                                                                 training_labels,
                                                                 testing_data,
                                                                 testing_labels)

                with open("results/" + CLASSIFIER_TYPE + "_parameter_resuts_"
                          + str(samples_per_class)
                          + "_samples_per_class.txt", "w") as outfile:
                    json.dump(acc_dic, outfile)

    elif CLASSIFIER_TYPE == "rnn":

        for lr in LR_LIST:

            net = RNNModel()

            acc_dic["lr_" + str(lr)], _ = train_net(net, training_data,
                                                    training_labels,
                                                    testing_data, testing_labels,
                                                    lr, samples_per_class)

            with open("results/" + CLASSIFIER_TYPE + "_parameter_resuts_"
                      + str(samples_per_class)
                      + "_samples_per_class.txt", "w") as outfile:
                json.dump(acc_dic, outfile)


if __name__ == "__main__":

    if os.path.exists("results/") is False:
        try:
            os.mkdir("results/")
        except OSError:
            pass

    print("\nUsing a " + CLASSIFIER_TYPE + " classifier")

    if BLOCKED_TRAINING_EXAMPLES_BOOL is True:

        assert len(SAMPLES_PER_CLASS_LIST) == 1, "Only provide one value for" \
            "samples-per-class for the CL experimental set-up"
        samples_per_class = SAMPLES_PER_CLASS_LIST[0]

        training_data, training_labels, testing_data, testing_labels = generic_setup(
            samples_per_class)

        blocked_training(training_data, training_labels, testing_data,
                         testing_labels, samples_per_class)

    else:

        for samples_per_class in SAMPLES_PER_CLASS_LIST:

            training_data, training_labels, testing_data, testing_labels = \
                generic_setup(samples_per_class)

            run_over_hyperparameters(training_data, training_labels,
                                     testing_data, testing_labels)
