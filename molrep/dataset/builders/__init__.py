#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy, inc.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


from omegaconf import OmegaConf

from molrep.dataset.builders.base_builder import BaseDatasetBuilder
from molrep.dataset.builders.property_builder import PropertyPredictionBuilder
from molrep.dataset.builders.pretraining_builder import PretrainingBuilder

from molrep.common.registry import registry

__all__ = [
    'BaseDatasetBuilder',
    'PropertyPredictionBuilder',
    'PretrainingBuilder',
]

def load_dataset(name, cfg_path=None):
    """
    Example
    >>> dataset = load_dataset("ogbg-molbbbp", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])
    """

    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    dataset, _ = builder.build_datasets()
    return dataset

def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            dataset_type: list(dataset_names.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items()) for dataset_type, dataset_names in v.DATASET_CONFIG_DICT.items()
        }


    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()