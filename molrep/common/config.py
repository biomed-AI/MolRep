#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""

from omegaconf import OmegaConf
from molrep.common.registry import registry


class Config:
    def __init__(self, args):
        self.config = {}
        self.args = args

        # Register the config and configuration for setup
        registry.register("configuration", self)

        config = OmegaConf.load(self.args.cfg_path)
        user_config = self._build_opt_list(self.args.options)

        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config)

        # Override the default configuration with user options.
        self.config = OmegaConf.merge(
            runner_config, model_config, dataset_config, user_config
        )

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def build_model_config(config, **kwargs):
        model = config.get("model", None)
        assert model is not None, "Missing model configuration file."

        model_cls = registry.get_model_class(model.name)
        assert model_cls is not None, f"Model '{model.name}' has not been registered."

        model_type = kwargs.get("model_type", None)
        if not model_type:
            model_type = model.get("model_type", 'default')
        model_config_path = model_cls.default_config_path(model_type=model_type)

        model_config = OmegaConf.create()
        # hiararchy override, customized config > default config
        model_config = OmegaConf.merge(
            model_config,
            OmegaConf.load(model_config_path),
            {"model": config["model"]},
        )
        return model_config

    @staticmethod
    def build_runner_config(config):
        runs = config.get("run", None)
        if runs is None:
            raise KeyError(
                "Expecting 'run' as the root key for dataset configuration."
            )

        run_cfg = OmegaConf.create()
        task_type = runs.get("task", None)
        assert task_type is not None, "Missing task type of running."

        builder_cls = registry.get_builder_class("base")
        run_config_path = builder_cls.default_task_config_path(task=task_type)
        default_config = {"run": OmegaConf.load(run_config_path)["run"]}

        # hiararchy override, customized config > default config
        run_config = OmegaConf.merge(
            run_cfg,
            default_config,
            {"run": config["run"]},
        )
        return run_config

    @staticmethod
    def build_dataset_config(config):
        datasets = config.get("datasets", None)
        if datasets is None:
            raise KeyError(
                "Expecting 'datasets' as the root key for dataset configuration."
            )

        dataset_config = OmegaConf.create()

        builder_cls = registry.get_builder_class("base")
        name = datasets.get("name", None)
        task = datasets.get("task_type", None)
        assert name is not None and task is not None, "Missing dataset name or type."
        dataset_config_path = builder_cls.default_config_path(task=task, name=name)
        default_config = {"datasets": OmegaConf.load(dataset_config_path)[name]}

        # hiararchy override, customized config > default config
        dataset_config = OmegaConf.merge(
            dataset_config,
            default_config,
            {"datasets": config["datasets"]},
        )
        return dataset_config

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def get_config(self):
        return self.config

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model

    @property
    def explainer_cfg(self):
        return self.config.run.explainer

    def to_dict(self):
        return OmegaConf.to_container(self.config)


def node_to_dict(node):
    return OmegaConf.to_container(node)