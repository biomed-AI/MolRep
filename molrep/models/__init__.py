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
from molrep.common.registry import registry

from molrep.models.sequence_based.MAT import MAT
from molrep.models.sequence_based.CoMPT import CoMPT
from molrep.models.sequence_based.BiLSTM import BiLSTM
from molrep.models.sequence_based.SALSTM import SALSTM
from molrep.models.sequence_based.Transformer import Transformer

from molrep.models.graph_learning.GIN import GIN
from molrep.models.graph_learning.ECC import ECC
from molrep.models.graph_learning.GAT import GAT
from molrep.models.graph_learning.DGCNN import DGCNN
from molrep.models.graph_learning.DiffPool import DiffPool
from molrep.models.graph_learning.GraphSAGE import GraphSAGE
from molrep.models.graph_learning.GraphNet import GraphNet

from molrep.models.graph_learning.MPNN import MPNN
from molrep.models.graph_learning.CMPNN import CMPNN
from molrep.models.graph_learning.DMPNN import DMPNN

from molrep.models.unsupervised_based.VAE import VAE
from molrep.models.unsupervised_based.RandomForest import RandomForest
from molrep.models.unsupervised_based.XGboost import XGboost

from molrep.models.interactions.link_models.PLNLP.PLNLP import PLNLP
from molrep.models.interactions.link_models.CFLP.CFLP import CFLP

from molrep.models.optims import LinearWarmupStepLRScheduler, LinearWarmupCosineLRScheduler

__all__ = [
    "load_model",
    "MPNN", "DMPNN", "CMPNN",
    "GIN", "ECC", "GAT", "DGCNN",
    "DiffPool", "GraphSAGE", "GraphNet",
    'MAT', 'CoMPT', 'BiLSTM', 'SALSTM',
    'Transformer', 'RandomForest', 'XGboost',
    'PLNLP', 'CFLP'
]


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.
    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.
    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_config(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

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


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    """
    Load model and its related preprocessors.
    List all available models and types in registry:
    >>> from molrep.models import model_zoo
    >>> print(model_zoo)
    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
    Returns:
        model (torch.nn.Module): model.
        processors (dict): preprocessors for molecular inputs.
    """



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
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()