# Numenta research papers code and data
This repository is currently under construction and will include the source code for all scripts used on Numenta's published papers. 

Here is the current list of published papers:

### [Why Does the Neocortex Have Layers and Columns, A Theory of Learning the 3D Structure of the World  • Jeff Hawkins, Subutai Ahmad & Yuwei Cui • Neuroscience • Preprint of journal submission • 2017/07/12][10]
This paper proposes a network model composed of columns and layers that performs robust object learning and recognition. The model introduces a new feature to cortical columns, location information, which is represented relative to the object being sensed. Pairing sensory features with locations is a requirement for modeling objects and therefore must occur somewhere in the neocortex. We propose it occurs in every column in every region.
> [Sources][10_src]
---
### [Why Neurons Have Thousands of Synapses, A Theory of Sequence Memory in Neocortex • Jeff Hawkins & Subutai Ahmad • Neuroscience • Published in Frontiers in Neural Circuits Journal • 2016/03/30][9]
Foundational paper describing core HTM theory for sequence memory and its relationship to the neocortex. Written with a neuroscience perspective, the paper explains why neurons need so many synapses and how networks of neurons can form a powerful sequence learning mechanism.
> [Sources][9_src]
---
### [Continuous Online Sequence Learning with an Unsupervised Neural Network Model • Yuwei Cui, Subutai Ahmad & Jeff Hawkins • Machine learning • Published in Neural Computation, November 2016, Vol 2- No. 11 • 2016/11/01][8]
Analysis of HTM sequence memory applied to various sequence learning and prediction problems. Written with a machine learning perspective, the paper contains some comparisons to statistical and Deep Learning techniques.
> [Sources][8_src]
---
### [Unsupervised Real-Time Anomaly Detection for Streaming Data • Subutai Ahmad, Alexander Lavin, Scott Purdy, Zuha Agha • Machine learning • Published in Neurocomputing, June 2017 • 2017/06/02][7]
This paper, which appears in a special issue of Neurocomputing, demonstrates how Numenta's online sequence memory algorithm, Hierarchical Temporal Memory, meets the requirements necessary for real-time anomaly detection in streaming data. It also presents results using the Numenta Anomaly Benchmark (NAB), the first open-source benchmark designed for testing anomaly detection algorithms on streaming data.
> [Sources][7_src]
---
### [The HTM Spatial Pooler: A Neocortical Algorithm for Online Sparse Distributed Coding • Yuwei Cui, Subutai Ahmad & Jeff Hawkins • Neuroscience • Preprint of journal submission • 2017/02/16][6]
This paper describes an important component of HTM, the HTM spatial pooler, which is a neurally inspired algorithm that learns sparse distributed representations online. Written from a neuroscience perspective, the paper demonstrates key computational properties of HTM spatial pooler.
> [Sources][6_src]
---
### [How Do Neurons Operate on Sparse Distributed Representations? A Mathematical Theory of Sparsity, Neurons and Active Dendrites • Subutai Ahmad & Jeff Hawkins • Neuroscience • Preprint of journal submission • 2016/01/05][5]
This paper describes a mathematical model for quantifying the benefits and limitations of sparse representations in neurons and cortical networks.
> [Sources][5_src]
---
### [Properties of Sparse Distributed Representations and their Application To Hierarchical Temporal Memory • Subutai Ahmad & Jeff Hawkins • Neuroscience • Research Paper • 2015/03/25][4]
An earlier version of the above submission, this paper applies our mathematical model of sparse representations to practical HTM systems.
> [Sources][4_src]
---
### [Evaluating Real-time Anomaly Detection Algorithms - the Numenta Anomaly Benchmark • Alexander Lavin & Subutai Ahmad • Machine learning • Published conference paper • 2015/10/12][3]
14th IEEE ICMLA 2015 - This paper discusses how we should think about anomaly detection for streaming applications. It introduces a new open-source benchmark for detecting anomalies in real-time, time-series data.
> [Sources][3_src]
---
### [Encoding Data for HTM Systems • Scott Purdy • Machine learning • Research Paper • 2016/02/18][2]
Hierarchical Temporal Memory (HTM) is a biologically inspired machine intelligence technology that mimics the architecture and processes of the neocortex. In this white paper we describe how to encode data as Sparse Distributed Representations (SDRs) for use in HTM systems. We explain several existing encoders, which are available through the open source project called NuPIC, and we discuss requirements for creating encoders for new types of data.
> [Sources][2_src]
---
### [Porting HTM Models to the Heidelberg Neuromorphic Computing Platform • Sebastian Billaudelle & Subutai Ahmad • Neuroscience • Research Paper • 2015/05/08][1]
Recently there has been much interest in building custom hardware implementations of HTM systems. This paper discusses one such scenario, and shows how to port HTM algorithms to analog hardware platforms such as the one developed by the Human Brain Project.
> [Sources][1_src]



[1]: https://arxiv.org/abs/1505.02142
[1_src]: arxiv/porting_htm_models_to_the_heidelberg_neuromorphic_computing_platform
[2]: https://arxiv.org/abs/1602.05925
[2_src]: arxiv/encoding_data_for_htm_systems
[3]: https://arxiv.org/abs/1510.03336
[3_src]: arxiv/evaluating_real_time_anomaly_detection_algorithms
[4]: https://arxiv.org/abs/1503.07469
[4_src]: arxiv/properties_of_sparse_distributed_representations_and_their_application_to_hierarchical_temporal_memory
[5]: https://arxiv.org/abs/1601.00720
[5_src]: arxiv/how_do_neurons_operate_on_sparse_distributed_representations
[6]: http://www.biorxiv.org/content/early/2017/02/16/085035
[6_src]: biorxiv/the_htm_spatial_pooler
[7]: http://www.sciencedirect.com/science/article/pii/s0925231217309864
[7_src]: neurocomputing/unsupervised_real_time_anomaly_detection_for_streaming_data
[8]: http://www.mitpressjournals.org/doi/abs/10.1162/neco_a_00893#.wcej8ueri18
[8_src]: neural_computation/continuous_online_sequence_learning_with_an_unsupervised_neural_network_model
[9]: http://journal.frontiersin.org/article/10.3389/fncir.2016.00023/full
[9_src]: frontiers/why_neurons_have_thousands_of_synapses
[10]: http://www.biorxiv.org/content/early/2017/07/12/162263
[10_src]: biorxiv/why_does_the_neocortex_have_layers_and_columns

