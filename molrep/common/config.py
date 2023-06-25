#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


import torch
import os, yaml
from itertools import groupby

from omegaconf import OmegaConf
from molrep.common.registry import registry


class Config:
    def __init__(self, args=None, cfg_path=None):
        self.config = {}
        self.args = args

        # Register the config and configuration for setup
        registry.register("configuration", self)
        user_config = self._build_opt_list(self.args.options)
        user_config_dict = self._build_config_dict(user_config)

        if self.args.cfg_path is None or not os.path.exists(self.args.cfg_path):
            cfg_path = os.path.join(registry.get_path("repo_root"), "projects", "user_defined.yaml")
            with open(cfg_path, 'w') as outfile:
                yaml.dump(user_config_dict, outfile, default_flow_style=False)
            self.args.cfg_path = cfg_path

        config = OmegaConf.load(self.args.cfg_path)
        runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, **user_config_dict)
        dataset_config = self.build_dataset_config(config)

        # Override the default configuration with user options.
        self.config = OmegaConf.merge(
            runner_config, model_config, dataset_config, user_config
        )

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.to_container(OmegaConf.from_dotlist(opts_dot_list))

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
            {"model": kwargs["model"]},
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
        if dataset_config_path is not None:
            default_config = {"datasets": OmegaConf.load(dataset_config_path)[name]}
        else:
            default_config = {"datasets": {}}

        # hiararchy override, customized config > default config
        dataset_config = OmegaConf.merge(
            dataset_config,
            default_config,
            {"datasets": config["datasets"]},
        )
        return dataset_config

    @classmethod
    def build_best_model_configs(cls, cfg_path):
        config = OmegaConf.load(cfg_path)
        runner_config = cls.build_runner_config(config)
        model_config = cls.build_model_config(config)
        dataset_config = cls.build_dataset_config(config)

        # Override the default configuration with user options.
        config = OmegaConf.merge(
            {"model_cfg": model_config.model, "datasets_cfg": dataset_config.datasets, 'run_cfg': runner_config}
        )
        return config

    @classmethod
    def build_explainer_configs(cls, cfg_path):
        config = OmegaConf.load(cfg_path)
        config = OmegaConf.merge(
            {'explainer_cfg': config}
        )
        return config

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def _build_config_dict(self, opts):
        cfg_dict = {
            "model": {},
            "datasets": {},
            "run": {}
        }

        # model configs: model_name, and other model parameters
        model_name = opts.pop("model_name", None)
        if model_name is None or registry.get_model_class(model_name) is None:
            print(f"Missing Model name or Model has not been registered.")
            print(f"Using the default gnn_model: GIN.")
            model_name = 'gin'
        cfg_dict["model"]["name"] = model_name

        # dataset configs:
        try:
            # must included:
            cfg_dict["datasets"]["name"] = opts.pop("dataset_name")
            cfg_dict["datasets"]["storage"] = [opts.pop("dataset_path")]
            cfg_dict["datasets"]["task_type"] = opts.pop("task_type")
            cfg_dict["datasets"]["smiles_column"] = opts.pop("smiles_column")
            target_columns = opts.pop("target_columns")
            cfg_dict["datasets"]["target_columns"] = target_columns if type(target_columns) == list else [target_columns]
        except:
            raise print(f"Missing dataset configs: ['dataset_name', 'dataset_path', 'task_type', 'smiles_column', 'target_columns']. ")

        # use default if not included
        cfg_dict["datasets"]["task"] = opts.pop("task", "property_prediction")
        cfg_dict["datasets"]["split_type"] = opts.pop("split_type", "random")
        default_metric_type = "auc" if cfg_dict["datasets"]["task_type"] == "classification" else "rmse"
        cfg_dict["datasets"]["metric_type"] = opts.pop("metric_type", default_metric_type)

        cfg_dict["datasets"]["num_tasks"] = opts.pop("num_tasks", len(cfg_dict["datasets"]["target_columns"]))
        cfg_dict["datasets"]["dim_target"] = opts.pop("dim_target", len(cfg_dict["datasets"]["target_columns"]))
        cfg_dict["datasets"]["multiclass_num_classes"] = opts.pop("multiclass_num_classes", 1)
        cfg_dict["datasets"]["feature"] = opts.pop("feature", "full")

        # run configs:
        cfg_dict["run"]["task"] = cfg_dict["datasets"]["task"]
        cfg_dict["run"]["device"] = opts.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        return cfg_dict

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
        return self.config.run_cfg.explainer

    def to_dict(self):
        return OmegaConf.to_container(self.config)


def node_to_dict(node):
    return OmegaConf.to_container(node)