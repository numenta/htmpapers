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
Load sparse distributed representations (SDRs) genereated from a k-Winner
Take All convolutional neural network, and constructs "objects"
compatible with the Columns Plus object-recognition algorithm
"""

import numpy as np


def generate_image_objects(
        data_set, num_samples_per_class, objectWidth,  # noqa: N803
        locationModuleWidth, data_set_section="SDR_classifiers_training",  # noqa: N803
        sanity_check=None):

    print("Loading " + data_set_section + " data-set from " + data_set)
    input_data = np.load("training_and_testing_data/" + data_set + "_SDRs_"
                         + data_set_section + ".npy")
    labels = np.load("training_and_testing_data/" + data_set + "_labels_"
                     + data_set_section + ".npy")
    images = np.load("training_and_testing_data/" + data_set + "_images_"
                     + data_set_section + ".npy")

    input_data_samples = []
    label_samples = []
    training_image_samples = []

    if sanity_check == "one_class_training":
        print("\nAs a sanity check, training using only a single class")
        num_classes = 1
    else:
        num_classes = 10

    for mnist_iter in range(num_classes):
        indices = np.nonzero(labels == mnist_iter)

        # Get num_samples_per_class of each digit/class type
        input_data_samples.extend(input_data[indices][0: num_samples_per_class])
        label_samples.extend(labels[indices][0: num_samples_per_class])
        training_image_samples.extend(images[indices][0: num_samples_per_class])

    features_dic = {}
    feature_name = 0
    width_one = locationModuleWidth

    objects_list = []

    # Keep track of how many exampels of particular MNIST digits have come up; used
    # to name unique samples iteratively
    sample_counter = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0}

    for sample_iter in range(len(label_samples)):

        sample_temp = np.reshape(input_data_samples[sample_iter], (128, 5, 5))
        sample_features_list = []

        for width_iter in range(objectWidth):
            for height_iter in range(objectWidth):

                # Convert the SDRs into sparse arrays (i.e. just representing the
                # non-zero elements)
                feature_temp = sample_temp[:, width_iter, height_iter]
                indices = np.array(np.nonzero(feature_temp)[0], dtype="uint32")

                # The location of the feature as expected by the Columns Plus-style
                # object
                top = width_one * width_iter
                left = width_one * height_iter

                features_dic[str(feature_name)] = indices  # Name each feature uniquely

                sample_features_list.append({"width": width_one,
                                             "top": top,
                                             "height": width_one,
                                             "name": str(feature_name),
                                             "left": left
                                             })

                feature_name += 1

        objects_list.append({"features": sample_features_list,
                             "name": str(label_samples[sample_iter]) + "_"
                             + str(sample_counter[str(label_samples[sample_iter])])})

        sample_counter[str(label_samples[sample_iter])] += 1

    print("Number of samples for each class ")
    print(sample_counter)

    return features_dic, objects_list, training_image_samples


if __name__ == "__main__":

    generate_image_objects(
        data_set="mnist", numObjects=10,
        objectWidth=5, locationModuleWidth=50)
