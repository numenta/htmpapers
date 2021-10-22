To reproduce the different figures exploring various types of hyperparameter search in the context of dendrites networks:

All the configs for running the various hyperparameter searches are in the file `hyperparameter_search.py` .


1. For the exploration of the number of segments, k-winner and weights sparsity HP searches in a basic dendritic network with the centroid context approach, the config are the following:  

  - segment_search
  - kw_sparsity_search
  - kw_sparsity_search_50
  - w_sparsity_search_50

2. For the exploration of all the cross hyperparameter in the same type of  network as   but only on 10 tasks, the config is :
    - cross_search

3. For the hyperparameter exploration of the dendrite number with synaptic intelligence on feedforward and/or on dendrites, the config is in `si_centroid.py`.
  - si_centroid_hp_10

4. Once the data has been generated you just need to run the makefile using `make` which will automatically run the data parsing script (`analyze_results.py`) with the right input and then run the plot generating script (`hyperparameters_figures.py`).
You will need to run `make` twice. The first time will error out in the middle but that is expected (it's due to the fact that the plots are in the same python script and not in two separated files).
