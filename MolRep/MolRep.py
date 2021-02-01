# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie
'''

import os
import torch
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from MolRep.Utils.config_from_dict import Config, Grid, DatasetConfig

from MolRep.Evaluations.model_selection.KFoldSelection import KFoldSelector
from MolRep.Evaluations.model_selection.HoldOutSelection import HoldOutSelector

from MolRep.Evaluations.model_assessment.KFoldAssessment import KFoldAssessment
from MolRep.Evaluations.model_assessment.HoldOutAssessment import HoldOutAssessment

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
                      additional_info = None,
                      holdout_test_size = 0,
                      inner_processes = 1,
                      seed = 42,
                      config_dir = 'MolRep/Configs/',
                      datasets_dir = 'MolRep/Datasets/',
                      output_dir = 'Outputs/'
                      ):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert (dataset_name is not None or dataset_path is not None), 'Dataset must be provided!'

    config_file = os.path.join(config_dir, f'config_{model_name}.yml')
    model_configurations = Grid(config_file)
    model_configuration = Config(**model_configurations[0])

    data_dir = Path(datasets_dir).parent / f"Data"
    split_dir = Path(datasets_dir).parent / f"Splits"

    if dataset_name in ['QM7b', 'QM8', 'QM9', 'ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', \
                                    'HIV', 'PDBbind', 'BACE', 'BBBP', 'Tox21', 'SIDER', 'ClinTox']:
        if dataset_path is not None:
            assert "This dataset-name is in our Datatbase, you should change your dataset-name."
        dataset_configuration = DatasetConfig(dataset_name)
        dataset_configuration.set_dataset_full_path(os.path.join(datasets_dir, dataset_configuration['path']))
    else:
        data_stats = {
                    'name': dataset_name,
                    'path': dataset_path,
                    'smiles_column': smiles_column,
                    'target_columns': target_columns,
                    'task_type': task_type,
                    'metric_type': metric_type,
                    'split_type': split_type,
                    'additional_info': {key:additional_info[key] for key in additional_info.keys()}
                    }
        dataset_configuration = DatasetConfig(dataset_name, data_dict=data_stats)
    
    if split_type == 'specific' and holdout_test_size == 0:
        holdout_test_size = 0.1
    exp_path = os.path.join(output_dir, f'{model_configuration.exp_name}_{dataset_configuration.exp_name}_assessment')

    dataset = DatasetWrapper(dataset_config=dataset_configuration,
                             model_name=model_configuration.exp_name,
                             split_dir=split_dir, features_dir=data_dir,
                             outer_k=outer_k, inner_k=inner_k, seed=seed, holdout_test_size=holdout_test_size)

    if inner_k is not None:
        model_selector = KFoldSelector(folds=inner_k, max_processes=inner_processes)
    else:
        model_selector = HoldOutSelector(max_processes=inner_processes)

    return dataset_configuration, dataset, model_configurations, model_selector, exp_path

def construct_assesser(model_selector, exp_path, model_configurations, dataset_configuration,
                       outer_k=None, max_processes=2):
    if outer_k is not None:
        risk_assesser = KFoldAssessment(outer_k, model_selector, exp_path, model_configurations, dataset_configuration,
                                        outer_processes=max_processes)
    else:
        risk_assesser = HoldOutAssessment(model_selector, exp_path, model_configurations, dataset_configuration,
                                          max_processes=max_processes)
    return risk_assesser

def construct_dataloader(dataset,
                         outer_k=None,
                         inner_k=None):

    return DataLoaderWrapper(dataset, outer_k=outer_k, inner_k=inner_k)
