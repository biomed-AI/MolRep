#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy, inc.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


import os

from omegaconf import OmegaConf
from molrep.common.registry import registry
from molrep.data.datasets.base_dataset import MoleculeSampler


def load_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    return cfg


@registry.register_builder("base")
class BaseDatasetBuilder:

    DATASET_CONFIG_DICT = {
        "property_prediction": "molrep/configs/datasets/property_prediction.yaml",
    }

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.dataset_config = self.config.datasets_cfg
        self.dataset_name = self.dataset_config.name

        self.cache_root = registry.get_path("cache_root")

    @classmethod
    def default_config_path(cls, type="property_prediction"):
        return os.path.join(registry.get_path("repo_root"), cls.DATASET_CONFIG_DICT[type])

    def build_datasets(self):
        # download, split, etc...
        self._download_and_load_data()
        datasets = self.build()
        return datasets

    def _download_and_load_data(self):
        """
        Download data files if necessary.
        All the molecular datasets should have unified format.
        """
        pass

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.
        build() can be dataset-specific. Overwrite to customize.
        """
        raise NotImplementedError
    
    def dataloaders(self):
        """
        A property to get and create dataloaders by split just in need.
        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.
        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.
        Currently do not support multiple datasets for validation and test.
        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        raise NotImplementedError