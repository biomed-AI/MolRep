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

from molrep.dataset.datasets import *
from molrep.dataset.builders import *
from molrep.tasks import *
from molrep.models import *
from molrep.explainer import *
from molrep.processors import *
from molrep.experiments import *
from molrep.common.registry import registry
from molrep.common.config import Config

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("library_root", root_dir)

repo_root = os.path.dirname(root_dir)
registry.register_path("repo_root", repo_root)

cache_root = os.path.join(repo_root, "cache")
registry.register_path("cache_root", cache_root)

feature_root = os.path.join(repo_root, "features")
registry.register_path("features_root", feature_root)

split_root = os.path.join(repo_root, "splits")
registry.register_path("split_root", split_root)



def load_model(name, property_name, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.
    Args:
        name (str): name of the model.
        property_name (str): type of the property.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.
    Returns:
        model (torch.nn.Module): model.
    """

    model, _ = registry.get_model_class(name).from_pretrained(property_name=property_name, checkpoint=checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.
    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.
    Args:
        config (dict): preprocessor configs.
    Returns:
        processors (dict): preprocessors for molecular inputs.
        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    name = config.datasets_cfg.get("task", "property_prediction")
    processor = registry.get_builder_class(name)(config)
    return processor


def load_model_and_preprocess(name, property_name, is_eval=False, device="cpu", checkpoint=None):
    """
    Load model and its related preprocessors.
    List all available models and types in registry:
    >>> from molrep.models import model_zoo
    >>> print(model_zoo)
    Args:
        name (str): name of the model.
        property_name (str): type of the property.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
    Returns:
        model (torch.nn.Module): model.
        processors (dict): preprocessors for molecular inputs.
    """

    model, config = registry.get_model_class(name).from_pretrained(property_name=property_name, checkpoint=checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    mol_processor = load_preprocess(config=config)
    return model.to(device), mol_processor


class ModelZoo:
    """
    A utility class to create string representation of available model architectures and types.
    >>> from molrep.models import model_zoo
    >>> # list all available models
    >>> print(model_zoo)
    >>> # show total number of models
    >>> print(len(model_zoo))
    """

    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {', '.join(types)}"
                    for name, types in self.model_zoo.items()
                ]
            )
            + "=" * 50
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()


def load_explainer(explainer_name):
    """
    Using supported Explainer model.
    Args:
        explainer_name (str): name of the explainer.
        property_name (str): type of the property.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.
    """

    explainer_cls = registry.get_explainer_class(explainer_name.lower())
    cfg = Config.build_explainer_configs(cfg_path=explainer_cls.default_config_path())
    explainer = explainer_cls.from_config(cfg.explainer_cfg)
    return explainer


def expalin_and_visualize(explainer, model, data, device=None, **kwargs):
    '''
    
        >>> from IPython.display import SVG
        >>> SVG(expalin_and_visualize(explainer, model, data)[0])
    '''
    smiles_list = [d.smiles for d in data.dataset]
    atom_weights, bond_weights = explainer_data(data, explainer, model, device, **kwargs)
    atom_scores = preprocessing_attributions(smiles_list, atom_weights, bond_weights)
    return visualize(smiles_list, atom_scores, **kwargs)