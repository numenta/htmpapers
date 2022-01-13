# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
# ------------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm, colors
from matplotlib import pyplot as plt
import os
import pickle

from main.dendrites import DendriticLayerBase

from .base import HookManagerBase
import wandb


class PolicyVisualizationsHook(HookManagerBase):

    def init_data_collection(self):
        self.activations = []

    def activation_hook_fn(self, module, input, output):
        # output is (batch size x num_units x num_segments) matrix
        self.activations.append(output.data)

    def export_data(self):
        """Returns current data and reinitializes collection
        target - shape (batch_size, )
        activations - shape (batch_size, 1950, 10)
        """
        activations = torch.cat(self.activations, dim=0)
        self.init_data_collection()
        return {self.__class__.__name__: activations}

    @classmethod
    def consolidate_and_report(cls, data, epoch=None, local_save_path=None):
        """
        Accepts a dictionary where key is the task index
        and value is a list with one entry per step take

        Class method, requires data argument
        """
        cls.get_visualization(data, epoch, local_save_path)

    @classmethod
    def get_visualization(cls, data, epoch, local_save_path):
        raise NotImplementedError


class AverageSegmentActivationsHook(PolicyVisualizationsHook):

    @classmethod
    def get_visualization(
        cls,
        data,
        epoch=None,
        local_save_path=None,
        unit_to_plot=[0, 5, 10]
    ):
        """
        Returns a heatmap of dendrite activations for a single unit, plotted using
        matplotlib.
        :param data: dictionary mapping task_id to dendrite_activations, which are
                     3D torch tensor with shape (batch_size, num_units,
                     num_segments) in which entry b, i, j gives the
                     activation of the ith unit's jth dendrite segment for
                     example b
        :param unit_to_plot: index of the unit for which to plot dendrite activations;
                             plots activations of unit 0 by default
        """

        log_dict = {}
        num_segments = next(iter(data.values())).size(2)

        for unit in unit_to_plot:

            # Have to transpose, as plot expects a matrix (segments, tasks)
            avg_activations = torch.stack([
                data[task_id][:, unit, :].mean(dim=0)
                for task_id in sorted(data)
            ]).T.numpy()

            vmax = np.abs(avg_activations).max()
            vmin = -1.0 * vmax

            ax = plt.gca()
            ax.imshow(avg_activations, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
            plt.colorbar(
                cm.ScalarMappable(norm=colors.Normalize(-1, 1), cmap="coolwarm_r"),
                ax=ax, location="left",
                shrink=0.6, drawedges=False, ticks=[-1.0, 0.0, 1.0]
            )

            ax.set_xlabel("Task")
            ax.set_ylabel("Segment")
            ax.set_xticks(range(len(data)))
            ax.set_yticks(range(num_segments))

            plt.tight_layout()

            # Prepare to report it to wandb
            plt_name = f"average_segment_activations_{unit}"
            log_dict[plt_name] = wandb.Image(plt)

            # Save a local copy if required by user
            if local_save_path is not None and epoch is not None:
                plot_save_path = os.path.join(local_save_path, plt_name)
                os.makedirs(plot_save_path, exist_ok=True)
                plt.savefig(f"{os.path.join(plot_save_path, str(epoch))}.svg", dpi=300)
                np.save(
                    f"{os.path.join(plot_save_path, str(epoch))}.npy", avg_activations)

            # Clear plot before exiting function to avoid interference
            plt.clf()

        return log_dict

    def attach(self, network):
        """
        TODO: if layer is always one, remove lists, asserts and list handling methods
        """
        dendrite_layers = []

        for _, layer in network.named_modules():
            if isinstance(layer, DendriticLayerBase):
                dendrite_layers.append(layer.segments)

        assert len(dendrite_layers) == 1, f"Found {len(dendrite_layers)} layers"

        dendrite_layers[0].register_forward_hook(self.activation_hook_fn)

class HiddenActivationsPercentOnHook(PolicyVisualizationsHook):

    @classmethod
    def get_visualization(
        cls,
        data,
        epoch=None,
        local_save_path=None,
        num_units_to_plot=64
    ):
        """
        Returns a heatmap with shape (num_categories, num_units) where cell c, i gives
        the mean value of hidden activations for unit i over all given examples from
        category c.

        :param data: dictionary mapping task_id to activations. activations is a 2D
                            torch tensor with shape (batch_size, num_units) where
                            entry b, i gives the activation of unit i for example b
        :param num_units_to_plot: an integer which gives how many columns to show, for
                                ease of visualization; only the first num_units_to_plot
                                units are shown
        """
        hidden_activations = torch.stack([
            data[task_id][:, :num_units_to_plot].mean(dim=0)
            for task_id in sorted(data)
        ]).numpy()
        max_val = np.abs(hidden_activations).max()

        ax = plt.gca()
        ax.imshow(hidden_activations, cmap="Greens", vmin=0, vmax=max_val)
        plt.colorbar(
            cm.ScalarMappable(cmap="Greens"), ax=ax, location="top",
            shrink=0.5, ticks=[0.0, 0.5, 1.0], drawedges=False
        )

        ax.set_aspect(2.5)
        ax.set_xlabel("Hidden unit")
        ax.set_ylabel("Task")
        ax.get_yaxis().set_ticks(range(len(data)))

        plt.tight_layout()
        plt_name = "hidden_activations_percent_on"
        log_dict = {plt_name: wandb.Image(plt)}

        # Save a local copy if required by user
        if local_save_path is not None and epoch is not None:
            plot_save_path = os.path.join(local_save_path, plt_name)
            os.makedirs(plot_save_path, exist_ok=True)
            plt.savefig(f"{os.path.join(plot_save_path, str(epoch))}.svg", dpi=300)
            np.save(
                f"{os.path.join(plot_save_path, str(epoch))}.npy", hidden_activations)

        # Clear plot before exiting function to avoid interference
        plt.clf()

        return log_dict

    def attach(self, network):
        """
        TODO: if layer is always one, remove lists, asserts and list handling methods
        """
        dendrite_layers = []

        for name, layer in network.named_modules():
            if isinstance(layer, nn.Sequential) and "dendrite" in name:
                dendrite_layers.append(layer)

        assert len(dendrite_layers) == 1, f"Found {len(dendrite_layers)} layers"

        dendrite_layers[0].register_forward_hook(self.activation_hook_fn)


class CombinedSparseVizHook():

    hooks_classes = [
        AverageSegmentActivationsHook,
        HiddenActivationsPercentOnHook
    ]

    def __init__(self, network):
        self.hooks = [hook(network) for hook in self.hooks_classes]
        network.collect_hook_data = self.export_data

    def export_data(self):
        # Output of each hook is a tuple target, activations
        combined_hooks_data = {}
        for hook in self.hooks:
            combined_hooks_data.update(hook.export_data())
        return combined_hooks_data

    @classmethod
    def consolidate_and_report(cls, data, epoch=None, local_save_path=None):
        """
        Accepts a dictionary where key is the task index
        and value is a list with one entry per step take

        Class method, requires data argument
        """
        visualizations = {}
        for hook in cls.hooks_classes:
            visualizations.update(
                hook.get_visualization(data[hook.__name__], epoch, local_save_path)
            )
        return visualizations
