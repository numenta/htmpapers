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

import abc


class HookManagerBase(metaclass=abc.ABCMeta):
    """
    Requires:
    - assigning a function to collect_hook_data in the recipient network
    - attaching a hook to the recipient network
    - a class method called consolidate_and_report that executes an action
    based on the data reported
    """

    def __init__(self, network):
        network.collect_hook_data = self.export_data
        self.attach(network)
        self.init_data_collection()

    def init_data_collection(self):
        self.hook_data = []

    def export_data(self):
        """Returns current data and reinitializes collection"""
        data_to_export = self.hook_data
        self.init_data_collection()
        return data_to_export

    @abc.abstractmethod
    def attach(self, network):
        """
        Attach hook to network
        Example: network.register_forward_hook(self.forward_hook)
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def consolidate_and_report(cls, data):
        """
        Accepts a dictionary where key is the task index
        and value is a list with one entry per step take
        Each value has the same format as the return of export_data function

        Class method, requires data argument

        Returns a dictionary that can be incorporated into a regular log dict
        """
        raise NotImplementedError
