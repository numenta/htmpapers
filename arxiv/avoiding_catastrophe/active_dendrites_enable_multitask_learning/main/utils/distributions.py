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

# TODO: remove functions not used. Adapt all remaining ones to torch
# TODO: import from stable_baselines3, instead of redefining? Find out which is better

import torch


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return -self.neglogp(x)

    def get_shape(self):
        return self.flatparam().shape

    @property
    def shape(self):
        return self.get_shape()

    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])


class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def pdfromlatent(self, latent_vector, init_scale, init_bias):
        raise NotImplementedError

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return CategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return torch.int32


class MultiCategoricalPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec

    def pdclass(self):
        return MultiCategoricalPd

    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return torch.int32


class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return torch.float32


class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BernoulliPd

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return torch.int32


class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return torch.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return torch.nn.functional.softmax(self.logits)

    def neglogp(self, x):
        nlp = torch.nn.functional.cross_entropy(
            input=self.logits, target=x.squeeze(), reduction="none"
        )  # input = [N, [nprob]], target = [N], one_hot encoding is not needed
        return nlp.unsqueeze(
            1
        )  # keep the same output shape as the function sample and entropy

    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - torch.max(other.logits, axis=-1, keepdims=True)
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, axis=-1, keepdims=True)
        z1 = torch.sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), dim=-1)

    def sample(self):
        u = torch.rand(self.logits.shape, dtype=self.logits.dtype).to(
            self.logits.device
        )
        return torch.argmax(
            self.logits - torch.log(-torch.log(u)), dim=-1, keepdim=True
        )  # not really understand this formula

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space):
    """Select the type of probability distribution to use depending on the action space
    of the environment.

    Parameters
    ----------
    ac_space : Space
        Action space of the environment.

    Returns
    -------
    Pd
        Probability distribution object.

    """
    from gym import spaces

    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        print("Using DiagGaussianPd for policy.")
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        print("Using CategoricalPd for policy.")
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        print("Using MultiCategoricalPd for policy.")
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        print("Using BernoulliPd for policy.")
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError
