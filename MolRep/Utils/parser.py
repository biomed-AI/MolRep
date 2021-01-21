# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie
'''

import re
import os
import json
import yaml
import torch
import argparse
import pickle

from pathlib import Path
from MolRep.Utils.utils import *

def check_device(config):
    if re.match('^(cuda(:[0-9]+)?|cpu)$', config) is None:
        raise argparse.ArgumentTypeError(
            'Wrong device format: {}'.format(config)
        )

    if config != 'cpu':
        splited_device = config.split(':')

        if (not torch.cuda.is_available()) or \
                (len(splited_device) > 1 and
                    int(splited_device[1]) > torch.cuda.device_count()):
            raise argparse.ArgumentTypeError(
                'Wrong device: {} is not available'.format(config)
            )

    return config

def get_basic_configs():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Basic-Configuration')
    group.add_argument('--model_name',
                        type=str, required=False,
                        help='fileName of model class')
    group.add_argument('--dataset_name',
                        type=str, required=False,
                        help='fileName of dataset class')
    group.add_argument('--dataset_path',
                        type=str, required=False, default='./MolRep/Datasets',
                        help='where to load DataSet , filetype:.csv')
    group.add_argument('--model_config',
                        type=str, required=False, default='./MolRep/Configs',
                        help='hyperParameters config_path')
    group.add_argument('--features_dir',
                        type=str, required=False, default=None,
                        help='where to save Features , filetype:.p, .pt')
    group.add_argument('--split_dir',
                        type=str, required=False, default=None,
                        help='cross_val splits.json')
    # group.add_argument('--output_dir',
    #                     type=str, required=False, default='./Output',
    #                     help='where to save training log and results')
    group.add_argument('--k_fold',
                        type=int, required=False, default=None,
                        help='cross validation: Number of folds. Must be at least 2. If None, will choose train/valid/test.')
    group.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    group.add_argument('--seed', required=False, default=2020,
                        help='Seed')
    group.add_argument('--gpu',
                        type=int, required=False, default=None,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    group.add_argument('--reduplicate', required=False, action='store_true', default=False,
                        help='reduplicate work, if True, training with different seed.')
    group.add_argument('--reduplicate_seed', required=False, default=[0, 1, 10, 20, 50, 101, 188, 1024, 2020, 8888],
                        type=int, nargs='+',)
    group.add_argument('--save_model',
                        type=bool, required=False, default=True,
                        help='if True, saving model.')
    group.add_argument('--shuffle', required=False, default=True,
                        help='Shuffle for K-fold')
    group.add_argument('--grid_search',
                        action='store_true', default=False,
                        help='if True, training by Grid Search for multiple configurations')

    group = parser.add_argument_group('Dataset-Configuration')
    group.add_argument('--dim_features',
                        type=int, required=False, default=None,
                        help='dimension of input features.')
    group.add_argument('--output_size',
                        type=int, required=False, default=1,
                        help='dimension of output class.')
    group.add_argument('--max_num_nodes',
                        type=int, required=False, default=None,
                        help='dimension of output class.')
    group.add_argument('--features_scaling', type=bool, required=False, default=False,
                        help='features scaling.')

    group = parser.add_argument_group('Special-Configuration')
    group.add_argument('--mol2vec_model_path',
                        type=str, required=False, default='./models/unsupervised_based/mol2vec/model_300dim.pkl',
                        help='where to save mol2vec pretrained model, filetype:.pkl')
    # group.add_argument('--features_generator', type=str, nargs='*',
    #                     choices=get_available_features_generators())
    group.add_argument('--use_input_features', type=bool, required=False, default=False,
                        help='CMPNN.')
    group.add_argument('--atom_descriptors', type=str, required=False, default=None,
                        help='CMPNN.')

    basic_configs = parser.parse_args()

    # if basic_configs.output_dir is not None:
    #     logging_str = f'{basic_configs.dataset_name}_{basic_configs.model_name}_'
    #     logging_str += 'logging_grid_search.log' if basic_configs.grid_search else 'logging.log'
    #     basic_configs.log_file = os.path.join(basic_configs.output_dir, basic_configs.dataset_name, logging_str)
    #     basic_configs.results_file = os.path.join(basic_configs.output_dir, basic_configs.dataset_name, f'{basic_configs.dataset_name}_{basic_configs.model_name}.results')
    #     basic_configs.model_dir = Path(basic_configs.output_dir) / basic_configs.dataset_name / f"Model"
    #     basic_configs.model_path = basic_configs.model_dir / f"{basic_configs.model_name}.pt"

    #     create_dir_if_not_exists(basic_configs.model_dir)
    #     create_dir_if_not_exists(basic_configs.output_dir)
    #     create_dir_if_not_exists(os.path.join(basic_configs.output_dir, basic_configs.dataset_name))
    #     delete_file_if_exists(basic_configs.log_file)
    #     delete_file_if_exists(basic_configs.results_file)

    if basic_configs.gpu is not None:
        torch.cuda.set_device(basic_configs.gpu)
        torch.backends.cudnn.enabled = False

    basic_configs.model_config = Path(basic_configs.model_config) / f"config_{basic_configs.model_name}.yml"

    return basic_configs

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
