# Numenta research papers code and data
This repository contains reproducible code for selected Numenta papers. It is currently under construction and will eventually include the source code for all the scripts used in Numenta's papers.

### [Going Beyond the Point Neuron: Active Dendrites and Sparse Representations for Continual Learning][8]
In this paper we investigate how dendritic properties can add value to ANNs in the context of continual learning, an area where ANNs suffer from catastrophic forgetting 
> [Sources][8_src]

### [How Can We Be So Dense? The Benefits of Using Highly Sparse Representations][7]
In this paper we discuss inherent benefits of high dimensional sparse representations. We focus on robustness and sensitivity to interference. These are central issues with today’s neural network systems where even small and large perturbations can cause dramatic changes to a network’s output.
> [Sources][7_src]

### [Locations in the Neocortex: A Theory of Sensorimotor Object Recognition Using Cortical Grid Cells][5]
This paper provides an implementation for a location layer with grid-like modules that encode object-specific locations. This layer is incorpated into a network with an input layer and simulations show how the model can learn many complex objects and later infer which learned object is being sensed.
> [Sources][5_src]

### [A Theory of How Columns in the Neocortex Enable Learning the Structure of the World][1]
This paper proposes a network model composed of columns and layers that performs robust object learning and recognition. The model introduces a new feature to cortical columns, location information, which is represented relative to the object being sensed. Pairing sensory features with locations is a requirement for modeling objects and therefore must occur somewhere in the neocortex. We propose it occurs in every column in every region.
> [Sources][1_src]

### [The HTM Spatial Pooler – a neocortical algorithm for online sparse distributed coding][2]
This paper describes an important component of HTM, the HTM spatial pooler, which is a neurally inspired algorithm that learns sparse distributed representations online. Written from a neuroscience perspective, the paper demonstrates key computational properties of HTM spatial pooler.
> [Sources][2_src]

### [Evaluating Real-time Anomaly Detection Algorithms - the Numenta Anomaly Benchmark][3]
14th IEEE ICMLA 2015 - This paper discusses how we should think about anomaly detection for streaming applications. It introduces a new open-source benchmark for detecting anomalies in real-time, time-series data.
> [Sources][3_src]

### [Unsupervised Real-Time Anomaly Detection for Streaming Data][4]
This paper discusses the requirements necessary for real-time anomaly detection in streaming data, and demonstrates how Numenta's online sequence memory algorithm, HTM, meets those requirements. It presents detailed results using the Numenta Anomaly Benchmark (NAB), the first open-source benchmark designed for testing real-time anomaly detection algorithms.
> [Sources][4_src]

### [Why Neurons Have Thousands of Synapses, A Theory of Sequence Memory in Neocortex][6]
Foundational paper describing core HTM theory for sequence memory and its relationship to the neocortex. Written with a neuroscience perspective, the paper explains why neurons need so many synapses and how networks of neurons can form a powerful sequence learning mechanism.
> [Sources][6_src]

[1]: https://doi.org/10.3389/fncir.2017.00081
[1_src]: frontiers/a_theory_of_how_columns_in_the_neocortex_enable_learning_the_structure_of_the_world
[2]: https://www.frontiersin.org/articles/10.3389/fncom.2017.00111
[2_src]: frontiers/the_htm_spatial_pooler_a_neocortical_algorithm_for_online_sparse_distributed_coding
[3]: https://arxiv.org/abs/1510.03336
[3_src]: https://github.com/numenta/NAB
[4]: http://www.sciencedirect.com/science/article/pii/S0925231217309864
[4_src]: neurocomputing/unsupervised_real_time_anomaly_detection_for_streaming_data
[5]: https://doi.org/10.3389/fncir.2019.00022
[5_src]: frontiers/location_in_the_neocortex_a_theory_of_sensorimotor_object_recognition_using_cortical_grid_cells
[6]: http://journal.frontiersin.org/article/10.3389/fncir.2016.00023/full
[6_src]: frontiers/why_neurons_have_thousands_of_synapses
[7]: https://arxiv.org/abs/1903.11257
[7_src]: arxiv/how_can_we_be_so_dense
[8]: https://www.biorxiv.org/content/10.1101/2021.10.25.465651v1
[8_src]: biorxiv/going_beyond_the_point_neuron
