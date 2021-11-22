Purpose of the code
===============
The following implements a network using grid-cell computations that can perform basic object recognition on visual tasks such as the MNIST data-set. It builds on the architecture described in [Lewis et al 2019](https://www.frontiersin.org/articles/10.3389/fncir.2019.00022/full), enabling similar principles to be applied to image-based tasks. The repository can be used to generate the results found in our BMVC 2021 paper, [Grid Cell Path Integration For Movement-Based Visual Object Recognition](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1506.html).

The basic approach is to first train supplementary networks (a CNN and a decoder) and generate data that can support learning in the downstream classifiers, including the presented architecture (GridCellNet). As well as being used to train and evaluate GridCellNet,  sparse distributed representations (SDRs) extracted from images can also be fed to other classifiers (k-nearest neighbour and recurrent neural networks). As such, the first stage is to create these SDRs. In addition to evaluating the accuracy of several classifiers, there are a few additional evaluations that can be run, outlined below.

Getting Started
=================
- The order for running the evaluations is below. Note that each of these has associated hyper-parameters that can be specified in the respective file. Where in doubt, please refer to the BMVC paper re. hyper-parameter choices.
1) SDR_CNN.py : train a supervised CNN and use it to output the SDR representations for later use.
2) SDR_decoder.py : train an auto-encoder for visualing GridCellNet representations, and save the image data for later use by GridCellNet.
3) (optional) SDR_classifiers.py : evaluate supervised classifiers on the SDRs generated above 
- At this point it's worth noting that num_sensations_to_inference.py and visualise_GridCellNet_predictions.py cannot be run until data has been generated from a GridCellNet. To do so requires the use of Docker, as its implementation builds on legacy Python 2 code.

Moving On to GridCellNet
=================
- To get up and running, navigate to python2_htm_docker and run the following, adjusting the absolute path of docker_dir as appropriate:
```
docker pull numenta/htmpapers:doi_10.3389_fncir.2019.00022

docker run -v `pwd`/python2_htm_docker/docker_dir:/home/host_dir -it numenta/htmpapers:doi_10.3389_fncir.2019.00022 /bin/bash
```
- Once in the docker container, navigate to host_dir (you will see several other legacy directories and files outside of this that can be ignored); the data generated in the earlier steps is located in training_and_testing_data/ - note that data within host_dir in the docker and docker_dir on your local system is shared
- train_and_eval_GridCellNet.py can now be used to specify the desired hyperparamters for GridCellNet, and run to train and evaluate a network
- As well as results on classification performance, this will generate the file object_prediction_sequences.npy in prediction_data
- Navigating back to visual_recognition_grid_cells ('exit' to leave the Docker container), you can now run either of the following
	- num_sensations_to_inference.py plots the accuracy of the GridCellNet vs the number of sensations that have been performed (e.g. 'saccades')
	- visualise_GridCellNet_predictions.py outputs images depicting the representation of the GridCellNet (including its predictions) while it carries out inference

Advice on Replicating Data From Our BMVC Paper
=================
- To achieve numerically identical results to those presented in the paper, please make note of the following.
1) Results were generated across three independently run seed sets; these are seed_val = 1; 2; and 3 for all Python 3/PyTorch-based results (e.g. the RNN classifiers), and SEED1 = 10, SEED2 = 11; SEED1 = 12, SEED2 = 13; and SEED1 = 14, SEED2 = 15 for the Python 2 results (GridCellNet). The same seed set was used end-to-end for a given set of results, including in the training of the CNN for generating SDRs. For example, to generate all results for the second seed-set, one would train the SDR-generating CNN with seed_val=2, run the RNN classifiers with seed_val=2, and run GridCellNet classifiers with SEED1 = 12, SEED2 = 13.
3) When running SDR_classifiers.py for the main classification results, ensure that the simulation is run with SAMPLES_PER_CLASS_LIST = [1, 5, 10, 20], and the learning rates are set as specified below for the given settings. NB that only the result from a pre-specified learning rate (obtained in cross-validation) was actually considered/used.
	- Arbitrary sequence LSTM results: LR_LIST = [0.005, 0.01]
	- Fixed sequence LSTM results: LR_LIST = [0.002, 0.005, 0.01, 0.02]
4) When running train_and_eval_GridCellNet.py for the main classification results, simply run the GridCellNet individually for a given hyper-parameter setting at a time, e.g. THRESHOLD_LIST = [12], CLASS_THRESHOLDS_LIST = [0.7].
