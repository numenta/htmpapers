# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
Visualize all dendrite metrics on a randomly-initialized dendritic network to determine
if/how subnetworks are being selected during forward passes.
"""

import torch
from torch import nn
from torch.utils.data import Dataset

from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    DendriticAbsoluteMaxGate1d,
    DendriticGate1d,
    plot_dendrite_activations_by_unit,
    plot_dendrite_overlap_matrix,
    plot_entropy_distribution,
    plot_hidden_activations_by_unit,
    plot_mean_selected_activations,
    plot_overlap_scores_distribution,
    plot_percent_active_dendrites,
    plot_representation_overlap_distributions,
    plot_representation_overlap_matrix,
)
from nupic.research.frameworks.dendrites.routing import generate_context_vectors
from nupic.research.frameworks.vernon import SupervisedExperiment, mixins
from nupic.torch.modules import KWinners, SparseWeights


# ------ Dendrites experiment classes
class DendritesSupermaskExperiment(mixins.PlotRepresentationOverlap,
                                   mixins.PlotHiddenActivations,
                                   mixins.TrackRepresentationSparsity,
                                   mixins.PlotDendriteMetrics,
                                   mixins.RezeroWeights,
                                   SupervisedExperiment):
    pass


# ------ Network
class DendriticMLP(nn.Module):
    """
    A dendritic network which is similar to a MLP with a two hidden layers, except that
    activations are modified by dendrites. The context input to the network is used as
    input to the dendritic weights.
                    _____
                   |_____|    # Classifier layer, no dendrite input
                      ^
                      |
                  _________
    context -->  |_________|  # Second linear layer with dendrites
                      ^
                      |
                  _________
    context -->  |_________|  # First linear layer with dendrites
                      ^
                      |
                    input
    """
    def __init__(self, input_size, output_size, hidden_size, num_segments, dim_context,
                 sparsity, kw=False, relu=False,
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):

        # The nonlinearity can either be k-Winners or ReLU, but not both
        assert not (kw and relu)

        super().__init__()

        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dim_context = dim_context
        self.kw = kw
        self.relu = relu

        # Forward layers & k-winners
        self.dend1 = dendritic_layer_class(
            module=nn.Linear(input_size, hidden_size),
            num_segments=num_segments,
            dim_context=dim_context,
            module_sparsity=sparsity,
            dendrite_sparsity=sparsity
        )
        self.dend2 = dendritic_layer_class(
            module=nn.Linear(hidden_size, hidden_size),
            num_segments=num_segments,
            dim_context=dim_context,
            module_sparsity=sparsity,
            dendrite_sparsity=sparsity
        )

        if kw:
            self.kw1 = KWinners(n=hidden_size, percent_on=0.05, k_inference_factor=1.0,
                                boost_strength=0.0, boost_strength_factor=0.0)
            self.kw2 = KWinners(n=hidden_size, percent_on=0.05, k_inference_factor=1.0,
                                boost_strength=0.0, boost_strength_factor=0.0)
        if relu:
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

        # Final classifier layer
        self.classifier = SparseWeights(nn.Linear(hidden_size, output_size),
                                        sparsity=sparsity)

    def forward(self, x, context):
        output = self.dend1(x, context=context)
        output = self.kw1(output) if self.kw else output
        output = self.relu1(output) if self.relu else output

        output = self.dend2(output, context=context)
        output = self.kw2(output) if self.kw else output
        output = self.relu2(output) if self.relu else output

        output = self.classifier(output)
        return output


# ------ Dataset
class RandomContextDataset(Dataset):
    """
    In this dataset,
        - input features are sampled i.i.d. from U(-1, 1),
        - contexts are binary sparse vectors (each uniquely associated with a target),
        - targets are drawn i.i.d. from a categorical distribution.
    """
    def __init__(self, num_classes, num_examples, context_sparsity, input_dim,
                 context_dim, train=True):

        # Register attributes
        self.num_classes = num_classes
        self.num_examples_per_class = int(num_examples / num_classes)
        self.num_examples = self.num_examples_per_class * num_classes
        self.context_sparsity = context_sparsity
        self.input_dim = input_dim
        self.context_dim = context_dim

        # Generate random input vectors
        self.data = 2.0 * torch.rand((self.num_examples, input_dim)) - 1.0

        # Generate targets
        self.targets = [[class_id for n in range(self.num_examples_per_class)]
                        for class_id in range(num_classes)]
        self.targets = torch.tensor(self.targets).flatten()

        # Generate binary context vectors with the desired sparsity
        percent_on = 1.0 - context_sparsity
        self.contexts = generate_context_vectors(num_contexts=num_classes,
                                                 n_dim=context_dim,
                                                 percent_on=percent_on)
        assert (self.contexts.sum(dim=1) == int(percent_on * context_dim)).all()

        self.contexts = torch.repeat_interleave(self.contexts,
                                                repeats=self.num_examples_per_class,
                                                dim=0)

        assert self.data.size(0) == self.contexts.size(0) == self.targets.size(0)

    def __getitem__(self, idx):
        return self.data[idx, :], self.contexts[idx, :], self.targets[idx].item()

    def __len__(self):
        return self.data.size(0)


# ------ Functions to retrieve stats gathered by hooks
def plot_sparsity_metrics_from_hooks():
    results = {}

    # Here, "{name}" is the name of the module.
    for name, _, in_sparsity, out_sparsity in exp.output_hook_manager.get_statistics():
        results.update({f"in_sparsity/{name}": in_sparsity})
        results.update({f"out_sparsity/{name}": out_sparsity})

    return results


def plot_hidden_activations_from_hooks():
    results = {}

    # Here, "{name}" is the name of the module.
    for name, _, hidden_activations in exp.ha_hook.get_statistics():
        visual = plot_hidden_activations_by_unit(hidden_activations, exp.ha_targets)
        results.update({f"hidden_activations/{name}": visual})

    return results


def plot_representation_overlap_from_hooks():
    results = {}

    # Here, "{name}" is the name of the module.
    for name, _, activations in exp.ro_hook.get_statistics():

        targets = exp.ro_targets

        visual = plot_representation_overlap_matrix(activations, targets)
        results.update({f"representation_overlap_matrix/{name}": visual})
        visual_1, visual_2 = plot_representation_overlap_distributions(activations,
                                                                       targets)
        results.update({f"representation_overlap_inter/{name}": visual_1})
        results.update({f"representation_overlap_intra/{name}": visual_2})

    return results


def plot_dendrite_metrics_from_hooks():

    results = {}

    # Gather and plot the statistics.
    # Here, "{name}" is the name of the module.
    for name, _, activations, winners in exp.dendrite_hooks.get_statistics():

        # Each 'plot_func' will be applied to each module being tracked.
        for metric_name, plotting_args in exp.metric_args.items():

            # All of the defaults were set in `process_args`.
            plot_func = plotting_args["plot_func"]
            plot_args = plotting_args["plot_args"]
            max_samples_to_plot = plotting_args["max_samples_to_plot"]

            # Only use up the the max number of samples for plotting.
            targets = exp.targets[:max_samples_to_plot]
            activations = activations[:max_samples_to_plot]
            winners = winners[:max_samples_to_plot]

            visual = plot_func(activations, winners, targets, **plot_args)
            results.update({f"{metric_name}/{name}": visual})

    return results


# ------ Plotting functions
def _plot_percent_active_dendrites(activations, winners, targets, **kwargs):
    return plot_percent_active_dendrites(winners, targets, **kwargs)


def _plot_dendrite_overlap_matrix(activations, winners, targets, **kwargs):
    return plot_dendrite_overlap_matrix(winners, targets, **kwargs)


def _plot_overlap_scores_distribution(activations, winners, targets):
    return plot_overlap_scores_distribution(winners, targets)


def _plot_entropy_distribution(activations, winners, targets):
    return plot_entropy_distribution(winners, targets)


if __name__ == "__main__":

    # Config
    dendrites_supermask = dict(
        experiment_class=DendritesSupermaskExperiment,

        dataset_class=RandomContextDataset,
        dataset_args=dict(
            num_classes=100,
            num_examples=16384,
            context_sparsity=0.95,
            input_dim=2048,
            context_dim=2048,
        ),

        model_class=DendriticMLP,
        model_args=dict(
            input_size=2048,
            output_size=100,
            hidden_size=2048,
            num_segments=20,
            dim_context=2048,
            sparsity=0.95,
            kw=True,
            relu=False,
        ),

        # Tracking input & output sparsity
        track_input_sparsity_args=dict(
            include_modules=[AbsoluteMaxGatingDendriticLayer, KWinners, nn.ReLU]
        ),
        track_output_sparsity_args=dict(
            include_modules=[AbsoluteMaxGatingDendriticLayer, KWinners, nn.ReLU]
        ),

        # Dendrite metrics
        plot_dendrite_metrics_args=dict(
            include_modules=[DendriticGate1d, DendriticAbsoluteMaxGate1d],
            percent_active=dict(
                plot_func=_plot_percent_active_dendrites,
                plot_freq=1,
                plot_args=dict(annotate=False),
                max_samples_to_plot=5000
            ),
            mean_selected=dict(
                plot_func=plot_mean_selected_activations,
                plot_freq=1,
                plot_args=dict(annotate=False),
                max_samples_to_plot=5000
            ),
            overlap_matrix=dict(
                plot_func=_plot_dendrite_overlap_matrix,
                plot_freq=1,
                plot_args=dict(annotate=False),
                max_samples_to_plot=5000
            ),
            overlap_scores_dist=dict(
                plot_func=_plot_overlap_scores_distribution,
                plot_freq=1,
                max_samples_to_plot=5000
            ),
            entropy_dist=dict(
                plot_func=_plot_entropy_distribution,
                plot_freq=1,
                max_samples_to_plot=5000
            ),
            dendrite_activations_by_unit=dict(
                plot_func=plot_dendrite_activations_by_unit,
                plot_freq=1,
                max_samples_to_plot=5000
            )
        ),

        # Hidden activations
        plot_hidden_activations_args=dict(
            include_modules=[AbsoluteMaxGatingDendriticLayer, KWinners],
            plot_freq=1,
            max_samples_to_plot=5000
        ),

        # Representation overlap
        plot_representation_overlap_args=dict(
            include_modules=[nn.ReLU, KWinners],
            plot_freq=1,
            plot_args=dict(annotate=False),
            max_samples_to_plot=5000
        ),

        batch_size=1500,
        distributed=False,
        num_classes=10,
        val_batch_size=1500,

        # Optimizer won't be used, but is required for initializing experiment class
        optimizer_class=torch.optim.SGD,
        optimizer_args=dict(lr=0),
    )

    # Setup Vernon experiment class
    exp_class = dendrites_supermask["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(dendrites_supermask)

    hook_managers = [
        exp.dendrite_hooks,
        exp.ha_hook,
        exp.ro_hook,
        exp.input_hook_manager,
        exp.output_hook_manager
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate through the dataset just one time
    exp.pre_epoch()
    for x, context, y in exp.train_loader:

        x = x.to(device)
        context = context.to(device)
        y = y.to(device)

        # Start tracking
        for hook in hook_managers:
            hook.start_tracking()

        # Compute forward pass and compute loss
        pred = exp.model(x, context)
        loss = exp.error_loss(pred, y)
        loss.backward()

        # Stop tracking
        for hook in hook_managers:
            hook.stop_tracking()

    exp.post_epoch()

    # Retrieve metrics
    results = {}
    results.update(plot_sparsity_metrics_from_hooks())
    results.update(plot_dendrite_metrics_from_hooks())
    results.update(plot_hidden_activations_from_hooks())
    results.update(plot_representation_overlap_from_hooks())

    print("Successfully retrieved metric results for:")
    for key in results.keys():
        print(f"- {key}")
