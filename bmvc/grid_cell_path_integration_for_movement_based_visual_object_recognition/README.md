Purpose of the code
===============
The following implements a network using grid-cell computations that can perform basic object recognition on visual tasks such as the MNIST data-set. It builds on the architecture described in [Lewis et al 2019](https://www.frontiersin.org/articles/10.3389/fncir.2019.00022/full), enabling similar principles to be applied to image-based tasks. 

The basic approach is to first train supplementary networks (a CNN and a decoder) and generate data that can support learning in the proposed architecture (GridCellNet). As well as being used to train and evaluate GridCellNet,  sparse distributed representations (SDRs) extracted from images can also be fed to other classifiers (k-nearest neighbour and recurrent neural networks). As such, the first stage is to create these SDRs. In addition to evaluating the accuracy of several classifiers, there are a few additional evaluations that can be run, outlined below.

Getting Started
=================
- The order for running the evaluations is below. Note that each of these has associated hyper-parameters that can be specified in the respective file. 
1) SDR_CNN.py : train a supervised CNN and use it to output the SDR representations for later use.
2) SDR_decoder.py : train an auto-encoder for visualing GridCellNet representations, and save the image data for later use by GridCellNet.
3) (optional) SDR_classifiers.py : evaluate supervised classifiers on the SDRs generated above 
- At this point it's worth noting that num_sensations_to_inference.py and visualise_GridCellNet_predictions.py cannot be run until data has been generated from a GridCellNet. To do so requires the use of Docker, as its implementation builds on legacy Python 2 code.

Moving On to GridCellNet
=================
- To get up and running, navigate to python2_htm_docker and run the following, adjusting the absolute path of docker_dir as appropriate:
```
docker pull numenta/htmpapers:doi_10.3389_fncir.2019.00022

docker run -v ~/nta/nupic.research/projects/visual_recognition_grid_cells/python2_htm_docker/docker_dir:/home/host_dir -it numenta/htmpapers:doi_10.3389_fncir.2019.00022 /bin/bash
```
- Once in the docker container, navigate to host_dir (you will see several other legacy directories and files outside of this that can be ignored); the data generated in the earlier steps is located in training_and_testing_data/ - note that data within host_dir in the docker and docker_dir on your local system is shared
- train_and_eval_GridCellNet.py can now be used to specify the desired hyperparamters for GridCellNet, and run to train and evaluate a network
- As well as results on classification performance, this will generate the file object_prediction_sequences.npy in prediction_data
- Navigating back to visual_recognition_grid_cells ('exit' to leave the Docker container), you can now run either of the following
	- num_sensations_to_inference.py plots the accuracy of the GridCellNet vs the number of sensations that have been performed (e.g. 'saccades')
	- visualise_GridCellNet_predictions.py outputs images depicting the representation of the GridCellNet (including its predictions) while it carries out inference