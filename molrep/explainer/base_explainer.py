#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023, Sun Yat-sen Univeristy.
All rights reserved.

@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
'''


import os
import abc
from typing import Text

from omegaconf import OmegaConf
from molrep.common.registry import registry


@registry.register_explainer("base")
class BaseExplainer(abc.ABC):
    """Abstract class for an attribution technique."""

    name: Text
    sample_size: int  # Number of graphs to hold in memory per input.

    EXPLAINER_CONFIG_DICT = {
        "default": "",
    }

    @abc.abstractmethod
    def explain(self, data, model, **kwargs):
        """Compute GraphTuple with node and edges importances.
        Assumes that x (GraphTuple) has node and edge information as 2D arrays
        and the returned attribution will be a list of GraphsTuple, for each
        graph inside of x, with the same shape but with 1D node and edge arrays.
        Args:
          x: Input to get attributions for.
          model: model that gives gradients, predictions, activations, etc.
          task_index: index for task to focus attribution.
          batch_index: index for example to focus attribution.
        """
        raise NotImplementedError

    @classmethod
    def default_config_path(cls):
        return os.path.join(registry.get_path("library_root"), cls.EXPLAINER_CONFIG_DICT['default'])

    @classmethod
    def from_config(cls, cfg=None):
        return cls(OmegaConf.to_container(cfg))