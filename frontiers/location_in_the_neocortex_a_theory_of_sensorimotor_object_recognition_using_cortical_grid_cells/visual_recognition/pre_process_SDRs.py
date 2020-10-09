# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017-2018, Numenta, Inc.  Unless you have an agreement
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

"""
A shared experiment class for recognizing 2D objects by using path integration
of unions of locations that are specific to objects.
"""

import numpy as np

def generate_image_objects(numObjects, featuresPerObject, objectWidth,
                            numFeatures, featureDistribution=None):

	data_set = 'mnist'

	assert(featuresPerObject==objectWidth*objectWidth), "\nThe number of features per object and total object size do not match."
	assert(numFeatures==numObjects*objectWidth*objectWidth), "\nThe total number of features is inconsistent."

	#Import the SDRs (e.g. for the first two exmamples in MNIST), along with their labels
	training_data = np.load(data_set + '_SDRs_training.npy')
	training_labels = np.load(data_set + '_labels_training.npy')

	#print(np.shape(training_data))
	#print(np.shape(training_labels))
	training_data_samples = training_data[0:numObjects,:]
	trianing_labels_samples = training_labels[0:numObjects]
	# print(np.shape(training_data_samples))
	# print(training_data_samples)
	# print(trianing_labels_samples)


	features_dic = {}
	objectWidth = 5
	# locationModuleWidth = 10
	width_one = 10*2
	width_total = (objectWidth-1)*width_one
	#print(width_total)

	objects_list = []

	unique_name = 0

	for example_iter in range(len(trianing_labels_samples)):
		#print(np.shape(training_data_samples[example_iter]))
		example_temp = np.reshape(training_data_samples[example_iter], (64, 5, 5))
		#print(np.shape(example_temp))
		example_features_list = []

		for width_iter in range(objectWidth):
			for height_iter in range(objectWidth):
				# ???Reshape the flattendd SDRs to be in a 5x5 grid (check reshaping has worked by looking for 10% sparsity in each 64 array)


				# Note assigning unique names for the features doesn't really matter from my perspective; the name itself has no
				# significance to object recognition, and only affects which SDR is retrieved; therefore although it is less 
				# memory efficient, it does not create issues for object recognition

				#Convert the SDRs into sparse arrays (i.e. just representing the non-zero elements)
				feature_temp = example_temp[:, width_iter, height_iter]
				#indices = np.nonzero(feature_temp)
				indices = np.array(np.nonzero(feature_temp)[0], dtype="uint32")
				feature_name = str(unique_name)
				unique_name += 1
				top = width_one*width_iter
				left = width_one*height_iter

				features_dic[feature_name] = indices
				example_features_list.append({
					'width': width_one,
					'top': top,
					'height': width_one,
					'name': feature_name,
					'left': left
					})
				# print(example_features_list)
				#print(features_dic)
				# print(np.shape(feature_temp))
				# print(np.sum(feature_temp)/np.shape(example_temp)[0])

		objects_list.append({'features':example_features_list, 'name':str(trianing_labels_samples[example_iter]) + '_' + str(example_iter)})
		#objects_list.append({'features':example_features_list, 'name':str(trianing_labels_samples[example_iter])})
				#Give each unique SDR a feature 'name' (number's 0 through number of unique SDRs; may be safe to initially just assume the SDRs are unique)

	# print(objects_list)
	# print('\n\n')
	# print(objects_dic)

	return features_dic, objects_list

if __name__ == '__main__':

	numObjects=2

	generate_image_objects(numObjects=numObjects, featuresPerObject=5*5, objectWidth=5,
                            numFeatures=5*5*numObjects)

#Create the appropriate 5x5 objects for the MNIST digits


# [{'features': [{'width': 20, 'top': 20, 'height': 20, 'name': '0', 'left': 0}, 
#   {'width': 20, 'top': 0, 'height': 20, 'name': '0', 'left': 0}, {'width': 20, 'top': 0, 'height': 20, 'name': '0', 'left': 20}], 'name': '0'}, 

#   {'features': [{'width': 20, 'top': 0, 'height': 20, 'name': '0', 'left': 20}, {'width': 20, 'top': 0, 'height': 20, 'name': '2', 'left': 0}, 
#     {'width': 20, 'top': 20, 'height': 20, 'name': '0', 'left': 20}], 'name': '1'}]


# [{'features': [{'width': 20, 'top': 0, 'left': 0, 'name': '0', 'height': 20}, {'width': 20, 'top': 0, 'left': 20, 'name': '1', 'height': 20}, 
# 	{'width': 20, 'top': 0, 'left': 40, 'name': '2', 'height': 20}, {'width': 20, 'top': 0, 'left': 60, 'name': '3', 'height': 20}, 
# 	{'width': 20, 'top': 0, 'left': 80, 'name': '4', 'height': 20}, {'width': 20, 'top': 20, 'left': 0, 'name': '5', 'height': 20}, 
# 	{'width': 20, 'top': 20, 'left': 20, 'name': '6', 'height': 20}, {'width': 20, 'top': 20, 'left': 40, 'name': '7', 'height': 20}, 
# 	{'width': 20, 'top': 20, 'left': 60, 'name': '8', 'height': 20}, {'width': 20, 'top': 20, 'left': 80, 'name': '9', 'height': 20}, 
# 	{'width': 20, 'top': 40, 'left': 0, 'name': '10', 'height': 20}, {'width': 20, 'top': 40, 'left': 20, 'name': '11', 'height': 20}, 
# 	{'width': 20, 'top': 40, 'left': 40, 'name': '12', 'height': 20}, {'width': 20, 'top': 40, 'left': 60, 'name': '13', 'height': 20}, 
# 	{'width': 20, 'top': 40, 'left': 80, 'name': '14', 'height': 20}, {'width': 20, 'top': 60, 'left': 0, 'name': '15', 'height': 20}, 
# 	{'width': 20, 'top': 60, 'left': 20, 'name': '16', 'height': 20}, {'width': 20, 'top': 60, 'left': 40, 'name': '17', 'height': 20}, 
# 	{'width': 20, 'top': 60, 'left': 60, 'name': '18', 'height': 20}, {'width': 20, 'top': 60, 'left': 80, 'name': '19', 'height': 20}, 
# 	{'width': 20, 'top': 80, 'left': 0, 'name': '20', 'height': 20}, {'width': 20, 'top': 80, 'left': 20, 'name': '21', 'height': 20}, 
# 	{'width': 20, 'top': 80, 'left': 40, 'name': '22', 'height': 20}, {'width': 20, 'top': 80, 'left': 60, 'name': '23', 'height': 20}, 
# 	{'width': 20, 'top': 80, 'left': 80, 'name': '24', 'height': 20}], 'name': '5_0'}, 

# {'features': [{'width': 20, 'top': 0, 'left': 0, 'name': '25', 'height': 20}, {'width': 20, 'top': 0, 'left': 20, 'name': '26', 'height': 20}, {'width': 20, 'top': 0, 'left': 40, 'name': '27', 'height': 20}, {'width': 20, 'top': 0, 'left': 60, 'name': '28', 'height': 20}, {'width': 20, 'top': 0, 'left': 80, 'name': '29', 'height': 20}, {'width': 20, 'top': 20, 'left': 0, 'name': '30', 'height': 20}, {'width': 20, 'top': 20, 'left': 20, 'name': '31', 'height': 20}, {'width': 20, 'top': 20, 'left': 40, 'name': '32', 'height': 20}, {'width': 20, 'top': 20, 'left': 60, 'name': '33', 'height': 20}, {'width': 20, 'top': 20, 'left': 80, 'name': '34', 'height': 20}, {'width': 20, 'top': 40, 'left': 0, 'name': '35', 'height': 20}, {'width': 20, 'top': 40, 'left': 20, 'name': '36', 'height': 20}, {'width': 20, 'top': 40, 'left': 40, 'name': '37', 'height': 20}, {'width': 20, 'top': 40, 'left': 60, 'name': '38', 'height': 20}, {'width': 20, 'top': 40, 'left': 80, 'name': '39', 'height': 20}, {'width': 20, 'top': 60, 'left': 0, 'name': '40', 'height': 20}, {'width': 20, 'top': 60, 'left': 20, 'name': '41', 'height': 20}, {'width': 20, 'top': 60, 'left': 40, 'name': '42', 'height': 20}, {'width': 20, 'top': 60, 'left': 60, 'name': '43', 'height': 20}, {'width': 20, 'top': 60, 'left': 80, 'name': '44', 'height': 20}, {'width': 20, 'top': 80, 'left': 0, 'name': '45', 'height': 20}, {'width': 20, 'top': 80, 'left': 20, 'name': '46', 'height': 20}, {'width': 20, 'top': 80, 'left': 40, 'name': '47', 'height': 20}, {'width': 20, 'top': 80, 'left': 60, 'name': '48', 'height': 20}, {'width': 20, 'top': 80, 'left': 80, 'name': '49', 'height': 20}], 'name': '0_1'}]

