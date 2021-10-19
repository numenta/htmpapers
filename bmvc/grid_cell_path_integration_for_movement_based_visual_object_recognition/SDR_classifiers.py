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
This trains basic classifiers on SDRs derived from images
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(18)
np.random.seed(18)

DATASET = "mnist"  # Options are "mnist" or "fashion_mnist"; note "fashion_mnist" not
# fully supported in the current implementation
SAMPLES_PER_CLASS_LIST = [5]  # How many samples per class to use in training the
# classifier; can be provided as a single-value list (e.g. [5]), or multi-valued list,
# in which case a classifier will be trained for each number of sample classes
CLASSIFIER_TYPE = "knn"  # Options are "rnn" and "knn"
LOCATION_DATA_BOOL = False  # Concatenates the location of a feature (0:24) to the
# features themselves, and provides this expanded feature to the classifier;
# recommended for RNN comparison models
ARBITRARY_SDR_ORDER_BOOL = False  # Option to shuffle the order of the SDRs for each
# example; this is used to determine the robustness of a classifier to a stream of
# inputs of arbitrary order
USE_RASTER_BOOL = False  # Follow a fixed rastor sequence if True and
# arbitrary order set to False

# Hyperparameters for RNN
EPOCHS = 1
WEIGHT_DECAY = 0.001  # Recommend 0.001
BATCH_SIZE = 128  # Recommend 128
LR_LIST = [0.002, 0.005, 0.01, 0.02]  # Learning rates to try for the RNN; can specify
# just a single-value list if desired

# Hyperparameters for k-NN
KNN_PROGRESSIVE_SENSATIONS_BOOL = False  # Whether to evaluate the k-NN classifier
# where progressively more input points are given (from just 1 up to the maximum of 25)
# This provides an indication of how many sensations are needed before classification
# is robust
N_NEIGHBOURS_LIST = list(range(1, 2))  # Number of neighbours to try for the k-NN, e.g.
# list(range(1, 11))


class RNNModel(torch.nn.Module):
    """
    Basic RNN to test the learning of sequential inputs.
    """
    def __init__(self):
        super(RNNModel, self).__init__()

        self.hidden_size = 128
        self.num_layers = 1
        self.rnn = torch.nn.LSTM(input_size=128 + int(LOCATION_DATA_BOOL),
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, 10)

    def forward(self, x):

        hidden_t = torch.zeros(self.num_layers, np.shape(x)[0], self.hidden_size)
        cell_t = torch.zeros(self.num_layers, np.shape(x)[0], self.hidden_size)

        x = x.reshape(-1, 128 + int(LOCATION_DATA_BOOL), 5 * 5)
        x = torch.FloatTensor(np.moveaxis(x.numpy(), 2, 1))  # Swap feature and
        # sequence axis

        out, (hidden_t, cell_t) = self.rnn(x, (hidden_t, cell_t))
        out = out[:, -1, :]  # Take the final representation
        out = self.fc(out)

        return out


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

    return input_data_samples, label_samples


def shuffle_sdr_order(input_data_samples, random_indices):
    """
    Shuffles the order of the input SDRs (total of 25)
    """
    sdr_shuffled_input_data_samples = []

    feature_dim = 128 + int(LOCATION_DATA_BOOL)

    for image_iter in range(len(input_data_samples)):

        if ARBITRARY_SDR_ORDER_BOOL is True:

            np.random.shuffle(random_indices)  # Re-shuffle the SDRs for each image
            # Otherwise the same fixed sequence is used to re-order them

        temp_sdr_array = np.reshape(input_data_samples[image_iter], (128, 5 * 5))

        # Include the location of the feature as a part of the feature itself
        if LOCATION_DATA_BOOL is True:
            temp_sdr_array = np.concatenate(
                (temp_sdr_array, np.transpose(random_indices[:, None])), axis=0)

        random_sdr_array = temp_sdr_array[:, random_indices]
        sdr_shuffled_input_data_samples.append(np.reshape(random_sdr_array,
                                                          (feature_dim * 5 * 5)))

    return sdr_shuffled_input_data_samples


def truncate_sdr_samples(input_data_samples, truncation_point):
    """
    Truncate the input SDRs, so as to evaluate e.g. how well a k-NN performs when given
    only 3 out of the total 25 input features
    """
    truncated_input_data_samples = []

    for image_iter in range(len(input_data_samples)):

        temp_sdr_array = np.reshape(
            input_data_samples[image_iter], (128 + int(LOCATION_DATA_BOOL), 5 * 5))
        truncated_sdr_array = temp_sdr_array[:, 0:truncation_point + 1]
        truncated_input_data_samples.append(
            np.reshape(
                truncated_sdr_array,
                ((128 + int(LOCATION_DATA_BOOL)) * (truncation_point + 1))))

    return truncated_input_data_samples


def load_data(data_section, random_indices, samples_per_class=5, sanity_check=None,
              dataset=DATASET):

    input_data = np.load("python2_htm_docker/docker_dir/training_and_testing_data/"
                         + DATASET + "_SDRs_" + data_section + ".npy")
    labels = np.load("python2_htm_docker/docker_dir/training_and_testing_data/"
                     + DATASET + "_labels_" + data_section + ".npy")

    print("\nLoading data from " + data_section)

    input_data_samples, label_samples = sub_sample_classes(input_data, labels,
                                                           samples_per_class,
                                                           sanity_check=None)

    input_data_samples = shuffle_sdr_order(input_data_samples, random_indices)  # Note
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

    for truncation_iter in range(25):

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

    if os.path.exists("results/" + str(samples_per_class) + "_samples_plots") is False:
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
                + str(lr) + str(lr) + "_lr_losses.png")
    plt.clf()

    print("Finished Training")
    return testing_acc


def run_classifier(samples_per_class):

    # Note the same fixed, random sampling of the input is used across all examples
    # in both training and testing, unless ARBITRARY_SDR_ORDER_BOOL==True
    random_indices = np.arange(25)

    if USE_RASTER_BOOL is False:
        np.random.shuffle(random_indices)

    training_data, training_labels = load_data(data_section="SDR_classifiers_training",
                                               random_indices=random_indices,
                                               samples_per_class=samples_per_class,
                                               sanity_check=None)

    # Note unless specified otherwise, the full test-dataset is not used for
    # evaluation, as this would take too long for GridCellNet
    testing_data, testing_labels = load_data(data_section="SDR_classifiers_testing",
                                             random_indices=random_indices,
                                             samples_per_class=100,
                                             sanity_check=None)

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

                with open("results/knn_parameter_resuts_" + str(samples_per_class)
                          + "_samples_per_class.txt", "w") as outfile:
                    json.dump(acc_dic, outfile)

    elif CLASSIFIER_TYPE == "rnn":

        for lr in LR_LIST:

            net = RNNModel()

            acc_dic["lr_" + str(lr)] = train_net(net, training_data, training_labels,
                                                 testing_data, testing_labels, lr,
                                                 samples_per_class)

            with open("results/rnn_parameter_resuts_" + str(samples_per_class)
                      + "_samples_per_class.txt", "w") as outfile:
                json.dump(acc_dic, outfile)


if __name__ == "__main__":

    if os.path.exists("results/") is False:
        try:
            os.mkdir("results/")
        except OSError:
            pass

    print("\nUsing a " + CLASSIFIER_TYPE + " classifier")

    for samples_per_class in SAMPLES_PER_CLASS_LIST:

        run_classifier(samples_per_class)
