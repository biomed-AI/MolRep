# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie
'''

import random
import time
import os
import json
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch 
import torch.nn as nn

from rdkit import Chem
from typing import Callable, List, Tuple, Union, Any
from argparse import Namespace

from MolRep.Models.scalers import StandardScaler


def worker_init(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def filter_invalid_smiles(smiles_list):
    valid_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            valid_smiles_list.append(smiles)
    return valid_smiles_list

def json_to_df(json_path):
    with open(json_path) as f:
        json_content = json.load(f)
        df = pd.DataFrame(json_content)
    return df

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)
    os.removedirs(path)

def delete_file_if_exists(path_file):
    if os.path.exists(path_file) and os.path.isfile(path_file):
        os.remove(path_file)


def create_logger(configs):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt=f'%Y-%m-%d %H:%M:%S')

    log_file = configs.log_file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    return logger


def read_config_file(dict_or_filelike):
    if isinstance(dict_or_filelike, dict):
        return dict_or_filelike

    path = Path(dict_or_filelike)
    if path.suffix == ".json":
        return json.load(open(path, "r"))
    elif path.suffix in [".yaml", ".yml"]:
        return yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        return pickle.load(open(path, "rb"))

    raise ValueError("Only JSON, YaML and pickle files supported.")


def getDate():
    ret = time.strftime('%m-%d-%H:%M', time.localtime(time.time()))
    return ret

def initialize_results_file(configs, model_config, writing='w', logger=None):
    writing = 'a' if configs.reduplicate else writing
    fp = open(configs.results_file, writing)
    model_config_string = ''
    if not isinstance(model_config, dict):
        model_config = model_config.config_dict
    for idx, (attrname, value) in enumerate(model_config.items()):
        if (idx+1) % 3 == 0:
            model_config_string += "%s: %s\n" % (attrname, value)
            continue
        model_config_string += "%s: %s\t" % (attrname, value)
    fp.write(model_config_string)
    fp.write('\n\n')
    fp.close()

def write_results(configs, log_str):
    fp = open(configs.results_file, 'a')
    fp.write(log_str)
    fp.write('\n')
    fp.close()

def save_checkpoint(path: str,
                    model: nn.Module,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    args: Namespace = None):
    """
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    model: nn.Module,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    logger: logging.Logger = None):
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    debug = logger.info if logger is not None else print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    if current_args is not None:
        args = current_args

    # Build model
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            # debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if torch.cuda.is_available():
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data scaler and the features scaler.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return scaler, features_scaler


def load_args(path: str) -> Namespace:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The arguments Namespace that the model was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage)['args']
