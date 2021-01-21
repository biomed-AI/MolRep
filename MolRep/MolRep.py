# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie
'''

import os
import torch
import warnings
warnings.filterwarnings("ignore")

from MolRep.Utils.config_from_dict import Config, Grid, DatasetConfig

from MolRep.Evaluations.model_selection.KFoldSelection import KFoldSelector

from MolRep.Evaluations.DatasetWrapper import DatasetWrapper
from MolRep.Experiments.experiments import EndToEndExperiment

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from MolRep.Evaluations.DataloaderWrapper import DataLoaderWrapper

def construct_dataset(dataset_name,
                      model_name,
                      inner_k = None,
                      outer_k = None,
                      dataset_path = None,
                      smiles_column = None,
                      target_columns = [],
                      task_type = None,
                      metric_type = None,
                      split_type = None,
                      holdout_test_size = 0,
                      inner_processes = 1,
                      seed = 42,
                      output_dir = 'Outputs/'
                      ):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert (dataset_name is not None or dataset_path is not None), 'Dataset must be provided!'

    config_file = f'MolRep/Configs/config_{model_name}.yml'
    model_configurations = Grid(config_file)
    model_configuration = Config(**model_configurations[0])

    if dataset_name in ['QM7b', 'QM8', 'QM9', 'ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', \
                                    'HIV', 'PDBbind', 'BACE', 'BBBP', 'Tox21', 'SIDER', 'ClinTox']:
        data_stats = None
    else:
        data_stats = {
                    'name': dataset_name,
                    'path': dataset_path,
                    'smiles_column': smiles_column,
                    'target_columns': target_columns,
                    'task_type': task_type,
                    'metric_type': metric_type,
                    'split_type': split_type
                    }
    dataset_configuration = DatasetConfig(dataset_name, data_dict=data_stats)
    exp_path = os.path.join(output_dir, f'{model_configuration.exp_name}_{dataset_configuration.exp_name}_assessment')

    dataset = DatasetWrapper(dataset_config=dataset_configuration,
                             model_name=model_configuration.exp_name, kfold_class=KFold,
                             outer_k=outer_k, inner_k=inner_k, seed=seed, holdout_test_size=holdout_test_size)

    model_selector = KFoldSelector(folds=inner_k, max_processes=inner_processes)

    return dataset_configuration, dataset, model_configurations, model_selector, exp_path

def construct_dataloader(dataset,
                         outer_k=None,
                         inner_k=None):

    return DataLoaderWrapper(dataset, outer_k=outer_k, inner_k=inner_k)
