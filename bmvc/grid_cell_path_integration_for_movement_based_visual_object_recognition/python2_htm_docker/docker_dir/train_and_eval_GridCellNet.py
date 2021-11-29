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
Train and evaluate a network based on grid-cell path integration
to perform visual object recognition given a sequence of SDR feature representations
Much of this code is based on the doExperiment function under capacity_simulation.py
in the "location_in_the_neocortex_a_theory_of_sensorimotor_
object_recognition_using_cortical_grid_cells" repository.
"""

import json
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from GridCellNet_base import (
    PIUNCorticalColumnForVisualRecognition,
    PIUNExperimentForVisualRecognition,
)
from pre_process_SDRs import generate_image_objects

# Core parameters for model urnning
THRESHOLD_LIST = [16]  # The different dendritic thresholds to try using for the
# GridCellNet classifier. Note this determines the threshold for the connections
# from feature columns to grid cells. Adjusting the dendritic threshold for grid cells
# to feature columns appears less important for performance on generalization
# Recommend values in the range of 12:16 for learning with 1:20 examples per class
NUM_SAMPLES_PER_CLASS_LIST = [20]  # The number of examples per class to give during
# training, e.g. [1, 5, 10, 20], with each one run as a separate model. Note using
# number of examples per class of 100+ requires hours of training time
CLASS_THRESHOLDS_LIST = [0.3]  # Determines the relative activity threshold that needs
# to be exceeded for the system to classify the input as a particular class. Between
# 0 and 1; recommend smaller values when learning more examples.
INPUT_GRID_DIMENSION = 5  # Options are 4 or 5, depending on the kind of pre-processing
# of the input used i.e. 7x7 pixel patches on a 4x4 grid, or 5x5 grid extracted from
# the middle layer of a CNN

NUM_TEST_OBJECTS_PER_CLASS = 100  # The number of test objects per class to use during
# evaluation
DATASET = "mnist"  # Options are "mnist" or "fashion_mnist", although functionality
# for fashion-mnist may be incomplete
SEED1 = 10
SEED2 = 11
BLOCKED_TRAINING_BOOL = True  # True by default; if False, shuffle the order of the
# example objects provided during training, providing the interleved training
# preferred by e.g. RNNs that use back-propagation
BLOCK_SPLIT = None  # Provide integer for the number of classes, or if not desired,
# None; only use the first BLOCK_SPLIT number of classes during training in order to
# support evaluating continual learning; (e.g. if 5, classes 0:4 inclusive will be used
# during training, but all classes will be used in evaluation)

FALSE_MOTOR_INFORMATION = False  # Provide unaligned motor and sensory information at
# inference as an ablation to determine how reliant GridCellNet is on these being
# consistent

VISUALIZE_PREDICTIONS_BOOL = False  # Generate data to allow later visualization of
# the network's predictions during inference. NB execution takes much longer, and so
# it is generally advised that this be set to False unless visualization specifically
# needed.

# Setting up special conditions for model learning and evaluation
FIXED_TOUCH_SEQUENCE = None  # Use a random but fixed sequence when performing
# inference for every object; options are None or
# range(INPUT_GRID_DIMENSION*INPUT_GRID_DIMENSION) --> use the latter if a
# fixed sequence is desired
if FIXED_TOUCH_SEQUENCE is not None:
    random.shuffle(FIXED_TOUCH_SEQUENCE)

EVAL_ON_TRAINING_DATA_BOOL = False  # If True, assess *recall* of objects
# learned during training rather than generalization to new objects
if EVAL_ON_TRAINING_DATA_BOOL:
    assert NUM_SAMPLES_PER_CLASS_LIST[0] == NUM_TEST_OBJECTS_PER_CLASS, \
        "If evaluating recall, ensure the number" \
        "of training and test objects is the same"

FLIP_BITS_LIST = [0]  # Evaluate robustness of recall of examples from the training
# data-set to bit-wise noise applied to the feature-level sensory representations
# Set to [0] if no noise during evaluation of recall is desired. Note does not affect
# standard inference/generalization, only recall of memorized objects.


def object_learning_and_inference(
        EVAL_ON_TRAINING_DATA_BOOL,  # noqa: N803
        locationModuleWidth,
        feature_columns_to_grid_cells_threshold,
        grid_cells_to_feature_columns_threshold,
        bumpType,
        cellCoordinateOffsets,
        cellsPerColumn,
        activeColumnCount,
        columnCount,
        featuresPerObject,
        objectWidth,
        numModules,
        seed1,
        seed2,
        anchoringMethod,
        num_flip_bits):

    if seed1 is not None:
        print("Setting random seed")
        np.random.seed(seed1)

    if seed2 is not None:
        print("Setting random seed")
        random.seed(seed2)

    locationConfigs = []  # noqa: N806

    perModRange = float((90.0 if bumpType == "square" else 60.0)  # noqa: N806
                        / float(numModules))

    print("\nUsing L4 dendritic thresholds of "
          + str(feature_columns_to_grid_cells_threshold))
    print("Using L6 dendritic thresholds of "
          + str(int(
                math.ceil(grid_cells_to_feature_columns_threshold
                          * numModules))) + "\n")

    for i in range(numModules):
        orientation = (float(i) * perModRange) + (perModRange / 2.0)

        # Note these parameters are for the L6 layer (grid cells), some of which are
        # reused (below) for the *L4* layer
        config = {
            "cellsPerAxis": locationModuleWidth,
            "scale": 40.0,
            "orientation": np.radians(orientation),
            "activationThreshold": feature_columns_to_grid_cells_threshold,
            "initialPermanence": 1.0,
            "connectedPermanence": 0.5,
            "learningThreshold": feature_columns_to_grid_cells_threshold,
            "sampleSize": -1,    # during learning, setting this to -1 means max new
            # synapses = len(activeInput)
            "permanenceIncrement": 0.1,
            "permanenceDecrement": 0.0,
            "cellCoordinateOffsets": cellCoordinateOffsets,
            "anchoringMethod": anchoringMethod
        }

        locationConfigs.append(config)

    l4Overrides = {  # noqa: N806
        "initialPermanence": 1.0,
        "activationThreshold": int(
            math.ceil(grid_cells_to_feature_columns_threshold * numModules)),
        "reducedBasalThreshold": int(
            math.ceil(grid_cells_to_feature_columns_threshold * numModules)),
        "minThreshold": numModules,
        "sampleSize": numModules,
        "cellsPerColumn": cellsPerColumn,
        "columnCount": columnCount
    }

    allLocationsAreUnique = None  # noqa: N806

    print("Using " + str(num_samples_per_class) + " training objects per class")

    column = PIUNCorticalColumnForVisualRecognition(
        locationConfigs, L4Overrides=l4Overrides, bumpType=bumpType)

    train_features_dic, train_objects, object_images = generate_image_objects(
        DATASET, num_samples_per_class, objectWidth,
        locationModuleWidth, data_set_section="SDR_classifiers_training",
        block_split=BLOCK_SPLIT)

    ColumnPlusNet = PIUNExperimentForVisualRecognition(  # noqa: N806
        column, features_dic=train_features_dic)

    if EVAL_ON_TRAINING_DATA_BOOL is True:
        print("Evaluating recall of objects seen during training")
        test_features_dic, test_objects, object_images = (train_features_dic,
                                                          train_objects,
                                                          object_images)

    elif EVAL_ON_TRAINING_DATA_BOOL is False:
        print("Using novel (never seen during training) objects to evaluate accuracy")

        test_features_dic, test_objects, object_images = generate_image_objects(
            DATASET, NUM_TEST_OBJECTS_PER_CLASS, objectWidth,
            locationModuleWidth, data_set_section="SDR_classifiers_testing",
            block_split=None)

    currentLocsUnique = True  # noqa: N806

    training_order_list = range(len(train_objects))

    if BLOCKED_TRAINING_BOOL is False:

        print("\n Using a non-blocked (i.e. interleved) ordering"
              " of training examples during learning")
        random.shuffle(training_order_list)

    print("\nLearning objects")
    for learning_object_iter in training_order_list:  # noqa: N806
        print("Learning " + train_objects[learning_object_iter]["name"])
        start_learn = time.time()

        objLocsUnique = ColumnPlusNet.learnObject(  # noqa: N806
            train_objects[learning_object_iter])
        currentLocsUnique = currentLocsUnique and objLocsUnique  # noqa: N806

    numFailures = 0  # noqa: N806 # General failures on classificaiton, including
    # either converged to an incorrect label, or never converged
    numIncorrect = 0  # noqa: N806
    numNeverConverged = 0  # noqa: N806
    total_sensations = 0

    # Over-write the features dictionary with those that are being used for evaluation
    ColumnPlusNet.features = test_features_dic

    if EVAL_ON_TRAINING_DATA_BOOL:

        for objectDescription in test_objects:  # noqa: N806

            (numSensationsToInference,  # noqa: N806
                incorrect) = ColumnPlusNet.recallObjectWithRandomMovements(
                    objectDescription,
                    flip_bits=num_flip_bits)

            if numSensationsToInference is None:
                numFailures += 1
                numNeverConverged += incorrect["never_converged"]
                numIncorrect += incorrect["false_convergence"]
            else:
                print("numSensationsToInference:" + str(numSensationsToInference))
                total_sensations += numSensationsToInference    # Keep a running
                # tally of sensations needed on successful trials to calculate the
                # mean number of sensations needed for inference

        print("Number of test objects: " + str(NUM_TEST_OBJECTS_PER_CLASS * 10))
        accuracy = 100 * ((NUM_TEST_OBJECTS_PER_CLASS * 10 - numFailures)
                          / float(NUM_TEST_OBJECTS_PER_CLASS * 10))
        errors = 100 * numFailures / float(NUM_TEST_OBJECTS_PER_CLASS * 10)
        false_converging = 100 * numIncorrect / float(NUM_TEST_OBJECTS_PER_CLASS * 10)
        never_converged = (100 * numNeverConverged
                           / float(NUM_TEST_OBJECTS_PER_CLASS * 10))

        mean_sensations = None

        if (NUM_TEST_OBJECTS_PER_CLASS * 10 - numFailures) > 0:
            mean_sensations = total_sensations / float(NUM_TEST_OBJECTS_PER_CLASS
                                                       * 10 - numFailures)

        result = {
            "num_samples_per_class" : num_samples_per_class,
            "num_test_objects_per_class" : NUM_TEST_OBJECTS_PER_CLASS,
            "accuracy" : accuracy,
            "total_errors" : errors,
            "false_converging": false_converging,
            "never_converged" : never_converged,
            "mean_sensations" : mean_sensations
        }

    else:

        object_iter = 0
        object_prediction_sequences = []

        for objectDescription in test_objects:  # noqa: N806

            start_infer = time.time()
            (numSensationsToInference, incorrect, prediction_sequence,  # noqa: N806
                touch_sequence, classification_visualization,
                single_converged_step) = ColumnPlusNet.inferObjectWithRandomMovements(
                    objectDescription,
                    objectImage=object_images[object_iter],
                    cellsPerColumn=cellsPerColumn,
                    class_threshold=class_threshold,
                    inputGridDimension=INPUT_GRID_DIMENSION,
                    fixed_touch_sequence=FIXED_TOUCH_SEQUENCE,
                    false_motor_information=FALSE_MOTOR_INFORMATION,
                    visualize_predictions_bool=VISUALIZE_PREDICTIONS_BOOL)

            print("Time to learn object : " + str(time.time() - start_learn))

            # Visualise how classification progressed
            # Useful for hyperparameter tuning
            true_mask = (np.nonzero(classification_visualization["classified"]))
            false_mask = np.invert(true_mask)
            plt.scatter(np.array(classification_visualization["step"])[false_mask],
                        np.array(
                            classification_visualization["proportion"])[false_mask],
                        alpha=0.5, label="Incorrect", color="crimson")
            plt.scatter(np.array(classification_visualization["step"])[true_mask],
                        np.array(
                            classification_visualization["proportion"])[true_mask],
                        alpha=0.5, label="Correct", color="dodgerblue")
            plt.ylim(0, 1)
            plt.xlim(0, 25)

            # object_prediction_sequences is used by downstream programs to e.g.
            # visualise the predictions of the network
            object_prediction_sequences.append(
                {"name": objectDescription["name"],
                 "prediction_sequence": prediction_sequence,
                 "touch_sequence": touch_sequence,
                 "correctly_classified": numSensationsToInference is not None,
                 "numSensationsToInference": numSensationsToInference,
                 "numSensationsToSingleConverge": single_converged_step,
                 "ground_truth_image": object_images[object_iter]})

            object_iter += 1
            if numSensationsToInference is None:
                numFailures += 1
                numNeverConverged += incorrect["never_converged"]
                numIncorrect += incorrect["false_convergence"]
            else:
                print("numSensationsToInference:" + str(numSensationsToInference))
                total_sensations += numSensationsToInference    # Keep a running
                # tally of sensations needed on successful trials to calculate the
                # mean number of sensations needed for inference
            print("Time to infer object : " + str(time.time() - start_infer))

        print("Saving prediction sequences...")
        np.save("prediction_data/object_prediction_sequences",
                object_prediction_sequences, allow_pickle=True)

        mean_sensations = None

        print("Number of test objects: " + str(NUM_TEST_OBJECTS_PER_CLASS * 10))
        accuracy = 100 * ((NUM_TEST_OBJECTS_PER_CLASS * 10 - numFailures)
                          / float(NUM_TEST_OBJECTS_PER_CLASS * 10))
        errors = 100 * numFailures / float(NUM_TEST_OBJECTS_PER_CLASS * 10)
        false_converging = 100 * numIncorrect / float(NUM_TEST_OBJECTS_PER_CLASS * 10)
        never_converged = (100 * numNeverConverged
                           / float(NUM_TEST_OBJECTS_PER_CLASS * 10))
        if (NUM_TEST_OBJECTS_PER_CLASS * 10 - numFailures) > 0:
            mean_sensations = total_sensations / float(NUM_TEST_OBJECTS_PER_CLASS
                                                       * 10 - numFailures)
        allLocationsAreUnique = currentLocsUnique  # noqa: N806

        result = {
            "num_samples_per_class" : num_samples_per_class,
            "num_test_objects_per_class" : NUM_TEST_OBJECTS_PER_CLASS,
            "accuracy" : accuracy,
            "total_errors" : errors,
            "false_converging": false_converging,
            "never_converged" : never_converged,
            "mean_sensations" : mean_sensations,
            "allLocationsAreUnique" : allLocationsAreUnique
        }

    print(result)

    return result


def dir_setup():
    if os.path.exists("misclassified/") is False:
        try:
            os.mkdir("misclassified/")
        except OSError:
            pass

    if os.path.exists("correctly_classified/") is False:
        try:
            os.mkdir("correctly_classified/")
        except OSError:
            pass

    if os.path.exists("results/") is False:
        try:
            os.mkdir("results/")
        except OSError:
            pass

    if os.path.exists("prediction_data/") is False:
        try:
            os.mkdir("prediction_data/")
        except OSError:
            pass


if __name__ == "__main__":

    dir_setup()

    numOffsets = 2  # noqa: N806, N816 # Spreads out the activation of grid cells to
    # model uncertainty about location
    cellCoordinateOffsets = tuple([i * (0.998 / (numOffsets - 1))  # noqa: N806, N816
                                  + 0.001
                                  for i in range(numOffsets)])

    for training_objects_iter in range(len(NUM_SAMPLES_PER_CLASS_LIST)):

        num_samples_per_class = NUM_SAMPLES_PER_CLASS_LIST[training_objects_iter]

        for threshold_iter in range(len(THRESHOLD_LIST)):

            f_to_g_threshold = THRESHOLD_LIST[threshold_iter]

            for class_threshold_iter in CLASS_THRESHOLDS_LIST:

                class_threshold = class_threshold_iter

                all_results = {}

                flip_results = []

                for num_flip_bits in FLIP_BITS_LIST:

                    all_results["class_threshold_"
                                + str(class_threshold)] = object_learning_and_inference(
                        EVAL_ON_TRAINING_DATA_BOOL=EVAL_ON_TRAINING_DATA_BOOL,
                        locationModuleWidth=50,  # This is the main parameter to
                        # consider increasing for challenging generalization tasks
                        # (e.g. learning many objects)
                        feature_columns_to_grid_cells_threshold=f_to_g_threshold,
                        # dendritic threshold for connections from the cortical
                        # columns representing input features, synapsing on the
                        # grid cells
                        grid_cells_to_feature_columns_threshold=0.9,
                        bumpType="square",  # No significance of using this or vs
                        # e.g. Gaussian bump for this task
                        cellsPerColumn=32,
                        columnCount=128,  # Total dimension of the input feature
                        # SDRs
                        activeColumnCount=29,  # Number of non-zero values in the
                        # input feature SDRs
                        cellCoordinateOffsets=cellCoordinateOffsets,
                        featuresPerObject=(INPUT_GRID_DIMENSION
                                           * INPUT_GRID_DIMENSION),
                        objectWidth=INPUT_GRID_DIMENSION,
                        numModules=40,  # 2nd parameter to increase (after
                        # locationModuleWidth) for greater representational
                        # capacity
                        seed1=SEED1,
                        seed2=SEED2,
                        anchoringMethod="corners",
                        num_flip_bits=num_flip_bits)

                    print(all_results)

                    if not EVAL_ON_TRAINING_DATA_BOOL:
                        plt.savefig("results/classification_visualization"
                                    "_plot__samples_per_class_"
                                    + str(num_samples_per_class)
                                    + "__dendritic_threshold_" + str(f_to_g_threshold)
                                    + "__class_threshold_"
                                    + str(class_threshold) + ".png")
                        plt.clf()

                    with open("results/GridCellNet_dendritic_threshold_"
                              + str(f_to_g_threshold) + "__class_threshold_"
                              + str(class_threshold)
                              + "__num_samples_per_class_"
                              + str(num_samples_per_class)
                              + "__bitwise_noise_" + str(num_flip_bits)
                              + "_all_results.json", "w") as outfile:
                        json.dump(all_results, outfile)

                    flip_results.append(all_results["class_threshold_"
                                        + str(class_threshold)]["accuracy"])

                if EVAL_ON_TRAINING_DATA_BOOL:

                    print("Final bit-wise noise results:")
                    print("Used dendritic threshold: " + str(f_to_g_threshold))
                    print("Bits flipped:")
                    print(FLIP_BITS_LIST)
                    print("Associated accuracy:")
                    print(flip_results)

                    plt.scatter(FLIP_BITS_LIST, flip_results)
                    plt.ylim(0, 100)
                    plt.xlim(0, 130)
                    plt.savefig("results/recall_vs_bitwise_noise__dendritic_threshold_"
                                + str(f_to_g_threshold) + ".png")
                    plt.clf()
