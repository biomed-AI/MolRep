#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


from molrep.common.registry import registry


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(cfg)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.
        Args:
            cfg (common.config.Config): _description_
        Returns:
            dict: List of torch.utils.data.Dataset objects by split.
        """

        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        name = datasets_config.get("task", "property_prediction")
        builder = registry.get_builder_class(name)(cfg)
        datasets, scaler = builder.build_datasets()

        return datasets, scaler

    def train_epoch(self, **kwargs):
        pass

    def evaluate_predictions(self, **kwargs):
        raise NotImplementedError

    def evaluation(self, **kwargs):
        pass

    def after_evaluation(self, **kwargs):
        pass
