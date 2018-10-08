# Numenta research papers code and data
This repository contains reproducible code for selected Numenta papers. It is currently under construction and will eventually include the source code for all the scripts used in Numenta's papers.

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

[1]: https://doi.org/10.3389/fncir.2017.00081
[1_src]: frontiers/a_theory_of_how_columns_in_the_neocortex_enable_learning_the_structure_of_the_world
[2]: https://www.frontiersin.org/articles/10.3389/fncom.2017.00111
[2_src]: frontiers/the_htm_spatial_pooler_a_neocortical_algorithm_for_online_sparse_distributed_coding
[3]: https://arxiv.org/abs/1510.03336
[3_src]: https://github.com/numenta/NAB
[4]: http://www.sciencedirect.com/science/article/pii/S0925231217309864
[4_src]: neurocomputing/unsupervised_real_time_anomaly_detection_for_streaming_data
[5]: https://doi.org/10.1101/436352
[5_src]: frontiers/location_in_the_neocortex_a_theory_of_sensorimotor_object_recognition_using_cortical_grid_cells
