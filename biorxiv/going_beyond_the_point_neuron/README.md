# Going Beyond the Point Neuron: Active Dendrites and Sparse Representations for Continual Learning

## Abstract

Biological neurons integrate their inputs on dendrites using a diverse range of non-linear functions. However the majority of artificial neural networks (ANNs) ignore biological neurons' structural complexity and instead use simplified point neurons. Can dendritic properties add value to ANNs? In this paper we investigate this question in the context of continual learning, an area where ANNs suffer from *catastrophic forgetting* (i.e., ANNs are unable to learn new information without erasing what they previously learned). We propose that dendritic properties can help neurons learn context specific patterns and invoke highly sparse context-specific subnetworks. Within a continual learning scenario, these task-specific subnetworks interfere minimally with each other and, as a result, the network remembers previous tasks significantly better than standard ANNs. We then show that by combining dendritic networks with Synaptic Intelligence (a biologically motivated method for complex weights) we can achieve significant resilience to catastrophic forgetting, more than either technique can achieve on its own. Our neuron model is directly inspired by the biophysics of sustained depolarization following dendritic NMDA spikes. Our research sheds light on how biological properties of neurons can be used to solve scenarios that are typically impossible for traditional ANNs to solve.

## Running experiments

This repository contains versioned code used to replicate the experiments in the paper. For the latest pytorch code for using the code in your own projects check the [nupic.research](https://github.com/numenta/nupic.research) and [nupic.torch](https://github.com/numenta/nupic.torch) repositories.

Below are instructions for reproducing all the charts and tables presented in the paper. There might be small differences due to randomness.

### Prerequisites

All the scripts in this directory were implemented using python 3.8. See [requirements.txt](./requirements.txt) for extra dependencies. 
When using anaconda virtual environment all you need to do is run the following command and conda will install everything for you. See environment.yml:

     conda env create

### Experiment configurations

Each individual experiment (which requires training a model on permutedMNIST for some number of tasks) has its own **experiment configuration**. An experiment configuration is simply a python dict defined in one of the files in `permutedMNIST/experiments/`. Each experiment configuration defines the model to train, its parameters and hyperparameters, the dataset, etc. For instance, the experiment configuration `PROTOTYPE_10` (defined in `permutedMNIST/experiments/prototype.py`) can be used to train an Active Dendrites Network on 10 permutedMNIST tasks while prototyping the context vector with the specified parameters.

Note that the experiment configuration contain the dataset class and arguments used to instantiate it, as below:

```
dataset_class=PermutedMNIST,
dataset_args=dict(
    root=os.path.expanduser("~/nta/results/data/"),
    download=False,  # Change to True if running for the first time
    seed=42,
),
```

To download the MNIST dataset (which can be used for continual learning on permutedMNIST), simply set the `download=True` in `dataset_args`. In most files in the folder `permutedMNIST/experiments/`, a "base configuration" is defined at the top of the file, and individual experiment configurations copy the base configuration.

### Training the models

All experiments can be run by executing the `run.py` file that exists in the folder `permutedMNIST/` and simultaneously specifying an experiment configuration. All experiment configurations are in the `permutedMNIST/experiments/` folder. For instance, the file `prototype.py` contains the configuration for training an Active Dendrites Network on 10 permutedMNIST tasks while prototyping a context vector. This can be done via

```
cd permutedMNIST
python run.py -e prototype_10
```

where the flag `-e` specifies the experiment configuration. We recommend using GPU acceleration. Otherwise some of the training will take significantly longer.

### List of all experiment configurations

Here we go in details about the configurations used to train a model for each of the results we described in the paper. We list configuration names, and each can be executed via the command `python run.py -e <configuration-name>` where the configuration name is one from below. In each configuration name, `*` can be replaced with the number of experiments: one of `2`, `5`, `10`, `25`, `50`, `100`. All configurations use the prototype method for computing the context vector (see Section 3.3 (Training method 1) in the paper).

  - `prototype_*` : the number of dendritic segments per neuron is equal to the number of continual learning tasks,
  - `prototype_*_segments_*` : the number of dendritic segments per neuron is always 10,
  - `si_prototype_*` : [Synaptic Intelligence (Zenke et al. (2017))](https://arxiv.org/abs/1703.04200) is used,
  - `active_dendrites_only_*` : ReLU is substituted for the [k-Winner-Take-All function (Ahmad & Scheinkman (2019))](https://arxiv.org/abs/1903.11257),
  - `sparse_representations_only_*` : neurons don't have dendritic segments.

The following configurations train a standard MLP in continual learning scenarios and `*` can only be substituted with `10` or `100`.

  - `three_layer_mlp_*` : MLP with 2 hidden layers (3 total layers),
  - `ten_layer_mlp_*` : MLP with 9 hidden layers (10 total layers).

## Figures

### Accuracy plots

The notebook `permutedMNIST/figures/figures.ipynb` gives accuracy plots from the paper, the source code that generated them, and the exact numerical results. Note that all numerical results in this notebook are averaged over 8 independent trials with different seed initializations. The experiment configurations to re-run each experiment are also provided in the `permutedMNIST/experiments/` folder.

### Subnetwork visualization

The notebook `permutedMNIST/figures/hidden_activations_per_task.ipynb` generates Figure 6 given hidden activation values in a trained Active Dendrites Network (on 10 permutedMNIST tasks). To generate these plots, do the following:

1. Use the config `HIDDEN_ACTIVATIONS_PER_TASK` to train an Active Dendrites Network: `python run.py -e hidden_activations_per_task` (from the `permutedMNIST/` folder).
2. The experiment, after completing, will generate a file of the form `x_10__activations.1_abcd.pt` and `y_10__activations.1_abcd.pt`. In the 2nd code block of the notebook, change the variable `key` so that it matches the randomly generated string in the name of the output file (i.e., it will probably be something other than `abcd`).
3. Run all code blocks in the notebook to generate a visualization similar to Figure 6.

----
"Going Beyond the Point Neuron: Active Dendrites and Sparse Representations for Continual Learning"; [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.25.465651v1).
