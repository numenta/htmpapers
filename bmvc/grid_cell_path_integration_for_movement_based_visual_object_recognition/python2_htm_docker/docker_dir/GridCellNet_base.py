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
Code to create column-networks with grid-cell representations that can
perform visual object recognition requiring generalization to unseen objects
"""

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
    ApicalTiebreakPairMemory,
)
from htmresearch.algorithms.location_modules import Superficial2DLocationModule
from htmresearch.frameworks.location.path_integration_union_narrowing import (
    PIUNCorticalColumn,
    PIUNExperiment,
)


class PIUNCorticalColumnForVisualRecognition(PIUNCorticalColumn):
    """
    A L4 + L6a network. Sensory input causes minicolumns in L4 to activate,
    which drives activity in L6a. Motor input causes L6a to perform path
    integration, updating its activity, which then depolarizes cells in L4.

    Whenever the sensor moves, call movementCompute. Whenever a sensory input
    arrives, call sensoryCompute.

    Adds method that provides an easily accessible list of the location
    representation across modules for copying, and over-writes initializaiton to
    correctly specify column cell dimensions.

    """
    def __init__(self, locationConfigs,  # noqa: N803
                 L4Overrides=None, bumpType="gaussian"):
        """
        @param L4Overrides (dict)
        Custom parameters for L4

        @param locationConfigs (sequence of dicts)
        Parameters for the location modules
        """
        self.bumpType = bumpType

        l4_cell_count = L4Overrides["columnCount"] * L4Overrides["cellsPerColumn"]

        if bumpType == "square":
            self.L6aModules = [
                Superficial2DLocationModule(
                    anchorInputSize=l4_cell_count,
                    **config)
                for config in locationConfigs]
        else:
            raise ValueError("Invalid bumpType", bumpType)

        l4_params = {
            "columnCount": 128,  # Note overriding below
            "cellsPerColumn": 32,
            "basalInputSize": sum(module.numberOfCells()
                                  for module in self.L6aModules)
        }

        if L4Overrides is not None:
            l4_params.update(L4Overrides)

        self.L4 = ApicalTiebreakPairMemory(**l4_params)

    def get_location_copy(self):
        """
        Get the population representation of the location layer for copying.
        """
        active_cells_list = []

        for module in self.L6aModules:

            active_cells_list.append(module.getActiveCells())

        return active_cells_list


class PIUNExperimentForVisualRecognition(PIUNExperiment):
    """
    An experiment class which passes sensory and motor inputs into a special two
    layer network and tracks the location of a sensor on an object.

    This version enables target classes to be used that evaluate generalization to
    unseen examples, as well as inputs derived from images such as MNIST.

    """

    def __init__(self, column,
                 features_dic=None,
                 noiseFactor=0,  # noqa: N803
                 moduleNoiseFactor=0,
                 num_grid_cells=40 * 50 * 50,
                 num_classes=10):
        """
        @param column (PIUNColumn)
        A two-layer network.

        @param featureNames (list)
        A list of the features that will ever occur in an object.

        Overwrite the original initializer to enable loading image-based features
        """
        self.column = column

        # Weights to learn associations between active locations and class labels
        self.class_weights = np.zeros((num_grid_cells, num_classes))

        # Use these for classifying SDRs and for testing whether they're correct.
        # Allow storing multiple representations, in case the experiment learns
        # multiple points on a single feature. (We could switch to indexing these by
        # objectName, featureIndex, coordinates.)
        # Example:
        # (objectName, featureIndex): [(0, 26, 54, 77, 101, ...), ...]
        self.locationRepresentations = defaultdict(list)
        self.inputRepresentations = {
            # Example:
            # (objectName, featureIndex, featureName): [0, 26, 54, 77, 101, ...]
        }

        # Load the set of features from the image-based data
        self.features = features_dic

        # For example:
        # [{"name": "Object 1",
        #   "features": [
        #       {"top": 40, "left": 40, "width": 10, "height" 10, "name": "A"},
        #       {"top": 80, "left": 80, "width": 10, "height" 10, "name": "B"}]]
        self.learnedObjects = []

        # The location of the sensor. For example: {"top": 20, "left": 20}
        self.locationOnObject = None

        self.maxSettlingTime = 10

        self.monitors = {}
        self.nextMonitorToken = 1

        self.noiseFactor = noiseFactor
        self.moduleNoiseFactor = moduleNoiseFactor

        self.representationSet = set()

    def learnObject(self,  # noqa: N802
                    objectDescription,  # noqa: N803
                    randomLocation=False,
                    useNoise=False,
                    noisyTrainingTime=1):
        """
        Train the network to recognize the specified object. Move the sensor to one of
        its features and activate a random location representation in the location
        layer. Move the sensor over the object, updating the location representation
        through path integration. At each point on the object, form reciprocal
        connections between the represention of the location and the representation
        of the sensory input.
        @param objectDescription (dict)
        For example:
        {"name": "Object 1",
         "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                      {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"}]}
        @return locationsAreUnique (bool)
        True if this object was assigned a unique set of locations. False if a
        location on this object has the same location representation as another
        location somewhere else.
        """
        self.reset()
        self.column.activateRandomLocation()

        locationsAreUnique = True  # noqa: N806
        all_locations = []

        if randomLocation or useNoise:
            numIters = noisyTrainingTime  # noqa: N806
        else:
            numIters = 1  # noqa: N806
        for _ in xrange(numIters):  # noqa: F821
            for iFeature, feature in enumerate(  # noqa: N806
                    objectDescription["features"]):
                self._move(feature, randomLocation=randomLocation, useNoise=useNoise)
                featureSDR = self.features[feature["name"]]  # noqa: N806
                self._sense(featureSDR, learn=True, waitForSettle=False)  # noqa: N806

                locationRepresentation = (  # noqa: N806
                    self.column.getSensoryAssociatedLocationRepresentation())
                self.locationRepresentations[(objectDescription["name"],
                                              iFeature)].append(locationRepresentation)
                self.inputRepresentations[(objectDescription["name"],
                                          iFeature, feature["name"])] = (
                                              self.column.L4.getWinnerCells())

                locationTuple = tuple(locationRepresentation)  # noqa: N806
                locationsAreUnique = (locationsAreUnique  # noqa: N806
                                      and locationTuple not in self.representationSet)

                # Track all the grid cells active over learning of an object
                all_locations.extend(locationRepresentation)

                self.representationSet.add(tuple(locationRepresentation))

            # Update the weights associating location reps with class labels
            unique_locations = list(set(all_locations))

            # Index by grid-cells that were active and the true class (i.e.
            # supervised signal)
            self.class_weights[unique_locations,
                               int(objectDescription["name"][0])] += 1

        self.learnedObjects.append(objectDescription)

        return locationsAreUnique

    def recallObjectWithRandomMovements(self,  # noqa: C901, N802
                                        objectDescription,  # noqa: N803
                                        flip_bits=0):  # noqa: N803
        """
        Attempt to recall the exact specified object (i.e. from the training data-set).
        Moves the sensor over the object until the object is recognized.
        NB this is essentially the same form of inference as used in the Lewis et
        al 2019 model.

        @param objectDescription (dict)
        For example:
        {"name": "Object 1",
         "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                      {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"}]}

        @param flip_bits (int)
        Number of bits to flip in the binary feature input array; used to evaluate
        the robustness of the system to sensory noise.

        @return inferredStep (int or None), incorrect (dic)
        """
        self.reset()

        inferred = False
        currentStep = 0  # noqa: N806
        incorrect = {"never_converged": 1, "false_convergence": 0}  # Track if the
        # non-recognition was due to convergance to an incorrect representation or
        # never converging

        # Generate touch sequence.
        touchSequence = range(len(objectDescription["features"]))  # noqa: N806
        random.shuffle(touchSequence)

        for iFeature in touchSequence:  # noqa: N806
            currentStep += 1
            feature = objectDescription["features"][iFeature]
            self._move(feature, randomLocation=False)

            featureSDR = self.features[feature["name"]]  # noqa: N806

            if flip_bits > 0:

                new_flipped_indices = random.sample(range(128), flip_bits)

                for to_flip in new_flipped_indices:

                    if to_flip in set(featureSDR):

                        # Remove the index if currently present (i.e flip a 1 to 0)
                        featureSDR = featureSDR[featureSDR != to_flip]  # noqa: N806

                    else:

                        featureSDR = np.append(featureSDR, to_flip)  # noqa: N806

                featureSDR = np.sort(featureSDR)  # noqa: N806

            self._sense(featureSDR, learn=False, waitForSettle=False)

            if not inferred:
                # Use the sensory-activated cells to detect whether the object has been
                # recognized. If these sensory-activated cells are correct, it implies
                # that the input layer's representation is classifiable -- the location
                # layer just correctly classified it.
                representation = \
                    self.column.getSensoryAssociatedLocationRepresentation()

                target_representations = set(np.concatenate(
                    self.locationRepresentations[
                        (objectDescription["name"], iFeature)]))

                inferred = ((set(representation) <= target_representations)
                            and (len(representation) > 0))

                if inferred:
                    print("Inferred!")
                    incorrect = {"never_converged": 0, "false_convergence": 0}
                    return currentStep, incorrect

                if not inferred and tuple(representation) in self.representationSet:
                    # We have converged to an incorrect representation
                    # - declare failure.
                    print("Converged to an incorrect representation!")
                    incorrect = {"never_converged": 0, "false_convergence": 1}
                    return None, incorrect

        if incorrect["never_converged"]:
            print("\nNever converged!")

        return None, incorrect

    def inferObjectWithRandomMovements(self,  # noqa: C901, N802
                                       objectDescription,  # noqa: N803
                                       objectImage,  # noqa: N803
                                       cellsPerColumn,  # noqa: N803
                                       class_threshold,
                                       inputGridDimension,
                                       fixed_touch_sequence=None,
                                       false_motor_information=False,
                                       visualize_predictions_bool=False):
        """
        Attempt to recognize the *class* of the specified object with the network.
        Moves the sensor over the object until the object is recognized.

        @param objectDescription (dict)
        For example:
        {"name": "Object 1",
         "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                      {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"}]}

        @param objectImage (Numpy array)
        The current object's image

        @param cellsPerColumn (int)
        Used to determine predicted columns.

        @param class_threshold (float)
        Determines the relative activity threshold that needs to be exceeded for the
        system to classify the input as a particular class. Between 0 and 1.

        @param fixed_touch_sequence (list or None)
        Use a random but fixed sequence when performing inference for every object.
        Used to evaluate the sensitivity of classification to the touch sequence
        employed across learning and inference.

        @param false_motor_information (bool)
        If True, provide movement information that does not correspond to the location
        of sensory inputs during inference.

        @param visualize_predictions_bool (bool)
        If True, iteratively run through inference to generate predictions of points
        that were not observed before the representation converged. The generated
        predictions can later be used to visualize how the classifier predicts
        unseen regions.

        @return inferredStep (int or None), incorrect (dic),
        prediction_sequence (list), touchSequence (list)
        """
        self.reset()

        inferredStep = None  # noqa: N806
        single_converged_step = None  # When the system representation has converged
        # to the size of a single previously learned object
        incorrect = {"never_converged": 1, "false_convergence": 0}  # Track if the
        # non-recognition was due to convergance to an incorrect representation or
        # never converging

        # Choose touch sequence.
        touchSequence = range(len(objectDescription["features"]))  # noqa: N806

        if fixed_touch_sequence is None:
            random.shuffle(touchSequence)
            print("\nPerforming inference using an arbitrary, unfixed"
                  " sequense of touches:")
            print(touchSequence)

        else:
            print("\nPerforming inference using a fixed random"
                  " sequense of touches:")
            touchSequence = fixed_touch_sequence  # noqa: N806
            print(touchSequence)

        # Touch sequence for motor input that *does not* align with sensed features
        false_touchSequence = range(len(objectDescription["features"]))  # noqa: N806
        # Only shuffle when actually using in order to preserve random seed behaviour
        if false_motor_information:
            random.shuffle(false_touchSequence)

        sense_sequence = []  # Contains a list of all the previous input SDRs and
        # (following convergence) predictions
        prediction_sequence = []  # Contains a list of lists of the previously sensed
        # and predicted features, where any given item represents the sense_sequence at
        # that touch iter in the overal inference process, followed by the
        # subsequent prediction

        # Hold useful data for visualizing when and why classification using a relative
        # activation threshold succeeds or fails
        classification_visualization = {}
        classification_visualization["classified"] = []
        classification_visualization["proportion"] = []
        classification_visualization["step"] = []

        additional_prediction_sensations = True  # Iteratively repeat sensation
        # sequence if convergence is successful, using the final sensation
        # to build up a prediction of what the network predicts in every part of the
        # image after its successfully converged to a single representation
        prediction_iter = 0

        while additional_prediction_sensations:

            print("Resetting network state")
            self.reset()
            currentStep = 0  # noqa: N806
            inferred = False
            single_converged = False  # True when location representation has converged
            # to that of a single object (i.e. location layer has a single object in
            # its representation, not a union of multiple possible objects)

            # Abort time-consuming generation of predictions after inference if
            # not needed
            if not visualize_predictions_bool:
                additional_prediction_sensations = False

            for i_feature in touchSequence:

                # Once converged to a single representation, progress through the
                # additional unsensed regions to predict what is present
                if single_converged:

                    if ((currentStep + prediction_iter)
                            == inputGridDimension * inputGridDimension):
                        additional_prediction_sensations = False
                        print("Reached end of predictions for this object.")
                        break

                    # Feature never sensed before convergence to single rep
                    # Ensures that visualization of predictions samples the entire
                    # unsensed space of the object.
                    i_feature = touchSequence[currentStep + prediction_iter]

                feature = objectDescription["features"][i_feature]

                # If the condition is desired, use an alternative, random
                # sequence of touches to derive movement information, which is
                # inconsistent with the sequence determining sensory inputs
                if false_motor_information:

                    false_move_i = false_touchSequence[currentStep]

                    self._move(objectDescription["features"][false_move_i])

                else:

                    self._move(feature)

                featureSDR = self.features[feature["name"]]  # noqa: N806

                self._sense(featureSDR, learn=False, waitForSettle=False)
                # Note re. visualizing predictions, _sense of the feature
                # itself does not inform the predicted columns on this
                # touch iteration (and thus does not invalidate the
                # prediction as a genuine prediction), but it does ensure the
                # BasalPredictedCells have been updated from the movement, and
                # we re-set the network state after performing each
                # post-convergance prediction

                predictedColumns = map(int, list(set(np.floor(  # noqa: N806
                    self.column.L4.getBasalPredictedCells() / cellsPerColumn))))

                currentStep += 1

                # Only collect the initial sensation sequence on the first pass
                if prediction_iter == 0:

                    # Include all previously sensed/predicted representaitons by
                    # over-writing current_sequence
                    current_sequence = sense_sequence[:]

                    current_sequence.append(list(predictedColumns))  # include
                    # the newly predicted columns

                    if currentStep == 1:  # On the first step, record the input
                        # sensation
                        prediction_sequence.append([featureSDR[:]])

                    else:
                        prediction_sequence.append(current_sequence)

                    # Record ground truth if not yet at converged representation
                    if not single_converged:
                        sense_sequence.append(featureSDR[:])

                    else:
                        # Once converged to a single representation has taken place,
                        # sense_sequence accumalates predictions

                        sense_sequence.append(list(predictedColumns))
                        prediction_iter += 1
                        break

                else:

                    if single_converged:

                        current_sequence = sense_sequence[:]

                        current_sequence.append(list(predictedColumns))  # include the
                        # newly predicted columns

                        prediction_sequence.append(current_sequence)

                        sense_sequence.append(list(predictedColumns))
                        prediction_iter += 1
                        break

                # Continue integrating sensory information to update location
                # representations until the object has both been inferred and the
                # location rep has converged to the size of a single object (which
                # enables visualization later by a decoder)
                if not single_converged:

                    # Use the sensory-activated location cells to detect whether
                    # the object has been recognized.
                    representation = \
                        self.column.getSensoryAssociatedLocationRepresentation()

                    max_active_proportion = 0.0  # Relative activation of the maximally
                    # active class neuron

                    if (len(representation) == 40) and (inferred is True):
                        # Handles the situation where inference proceeds single-rep
                        # convergence

                        single_converged = True
                        print("Inferred and now converged to a single representation")
                        single_converged_step = currentStep

                    if not inferred:

                        # NB a minimum number of steps are required before inference
                        # takes place; this reduces false positives in very early
                        # inference
                        if (len(set(representation)) > 0) and (currentStep >= 5):

                            # Vector to store 1 where a location has been active
                            active_loc_vector = np.zeros(
                                np.shape(self.class_weights)[0])

                            active_loc_vector[representation] = 1

                            class_node_activations = np.matmul(active_loc_vector,
                                                               self.class_weights)

                            # Track the proportion by which the most active node is
                            # firing, as well as whether it's the correct node
                            max_active_proportion = (np.max(class_node_activations)
                                                     / np.sum(class_node_activations))

                            # For later plotting of classification behaviour
                            # Useful in hyperparameter tuning
                            classification_visualization["classified"].append(
                                np.argmax(class_node_activations)
                                == int(objectDescription["name"][0]))
                            classification_visualization["proportion"].append(
                                max_active_proportion)
                            classification_visualization["step"].append(
                                currentStep)

                        inferred = (max_active_proportion >= class_threshold)

                        if inferred:
                            if np.argmax(class_node_activations) \
                                    == int(objectDescription["name"][0]):
                                print("Correctly classified "
                                      + objectDescription["name"][0])
                                inferredStep = currentStep  # noqa: N806
                                plt.imsave("correctly_classified/"
                                           + objectDescription["name"]
                                           + ".png", objectImage)
                                incorrect = {"never_converged": 0,
                                             "false_convergence": 0}

                                if len(representation) == 40:

                                    print("Converged to a single representation and"
                                          " now inferred")
                                    single_converged = True
                                    single_converged_step = currentStep

                            else:
                                print("Incorrectly classified a "
                                      + objectDescription["name"][0]
                                      + " as a "
                                      + str(np.argmax(class_node_activations)))
                                incorrect = {"never_converged": 0,
                                             "false_convergence": 1}
                                plt.imsave("misclassified/example_"
                                           + objectDescription["name"]
                                           + "_converged_to_"
                                           + str(np.argmax(class_node_activations))
                                           + ".png", objectImage)

                                return (None, incorrect, prediction_sequence,
                                        touchSequence, classification_visualization,
                                        None)

                if ((inferred and (not visualize_predictions_bool))
                        or (currentStep == inputGridDimension * inputGridDimension)):
                    additional_prediction_sensations = False
                    break

        if incorrect["never_converged"]:
            print("Never converged!")

        return (inferredStep, incorrect, prediction_sequence, touchSequence,
                classification_visualization, single_converged_step)
