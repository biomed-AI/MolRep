
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

# Model Selection
from MolRep.Evaluations.model_selection.KFoldSelection import KFoldSelector
from MolRep.Evaluations.model_selection.HoldOutSelection import HoldOutSelector

# Model Assessment
from MolRep.Evaluations.model_assessment.KFoldAssessment import KFoldAssessment
# from MolRep.Evaluations.model_assessment.HoldOutAssessment import HoldOutAssessment

from MolRep.Evaluations.DatasetWrapper import DatasetWrapper
from MolRep.Experiments.experiments import EndToEndExperiment

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

def endtoend(config_file, dataset_name, 
                outer_k, outer_processes, inner_k, inner_processes, 
                output_dir, datasets_dir, holdout_test_size=0.1, debug=False, data_stats=None):

    # Needed to avoid thread spawning, conflicts with multi-processing. You may set a number > 1 but take into account
    # the number of processes on the machine

    data_dir = Path(datasets_dir).parent / f"Data"
    split_dir = Path(datasets_dir).parent / f"Splits"


    torch.set_num_threads(1)
    outer_k = int(outer_k) if outer_k != 'None' else None
    inner_k = int(inner_k) if inner_k != 'None' else None

    experiment_class = EndToEndExperiment

    model_configurations = Grid(config_file)
    model_configuration = Config(**model_configurations[0])
    dataset_configuration = DatasetConfig(dataset_name, data_dict=data_stats)
    dataset_configuration.set_dataset_full_path(os.path.join(datasets_dir, dataset_configuration['path']))

    exp_path = os.path.join(output_dir, f'{model_configuration.exp_name}_{dataset_configuration.exp_name}_assessment')

    dataset = DatasetWrapper(dataset_config=dataset_configuration,
                             model_name=model_configuration.exp_name,
                             split_dir=split_dir, features_dir=data_dir,
                             outer_k=outer_k, inner_k=inner_k, holdout_test_size=holdout_test_size)
    model_selector = KFoldSelector(folds=inner_k, max_processes=inner_processes)
    risk_assesser = KFoldAssessment(outer_k, model_selector, exp_path, model_configurations, dataset_configuration,
                                    outer_processes=outer_processes)

    risk_assesser.risk_assessment(dataset, experiment_class, debug=debug)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config_file')
    parser.add_argument('--experiment', dest='experiment', default='endtoend')
    parser.add_argument('--result_folder', dest='result_folder', default='MolRep/Outputs/')
    parser.add_argument('--dataset_name', dest='dataset_name', default='none')
    parser.add_argument('--outer_folds', dest='outer_folds', default=10)
    parser.add_argument('--outer_processes', dest='outer_processes', type=int, default=3)
    parser.add_argument('--inner_folds', dest='inner_folds', default=5)
    parser.add_argument('--inner_processes', dest='inner_processes', type=int, default=1)
    parser.add_argument('--holdout_test_size', dest='holdout_test_size', default=0.1)
    parser.add_argument('--debug', action="store_true", dest='debug')

    parser.add_argument('--dataset_path', dest='dataset_path', default=None)
    parser.add_argument('--smiles_column', dest='smiles_column', default=None)
    parser.add_argument('--target_columns', dest='target_columns', default=[])
    parser.add_argument('--task_type', dest='task_type', default=None)
    parser.add_argument('--metric_type', dest='metric_type', default=None)
    parser.add_argument('--split_type', dest='split_type', default=None)

    args = parser.parse_args()

    config_file = args.config_file
    dataset_name = args.dataset_name
    experiment = args.experiment

    if args.dataset_name in ['QM7b', 'QM8', 'QM9', 'ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', \
                                    'HIV', 'PDBbind', 'BACE', 'BBBP', 'Tox21', 'SIDER', 'ClinTox', 'OS_cell']:
        endtoend(config_file, dataset_name,
                outer_k=args.outer_folds, outer_processes=args.outer_processes,
                inner_k=args.inner_folds, inner_processes=args.inner_processes, holdout_test_size=args.holdout_test_size,
                output_dir=args.result_folder, debug=args.debug)
    
    else:
        data_dict = {
                'name': args.dataset_name,
                'path': args.dataset_path,
                'smiles_column': args.smiles_column,
                'target_columns': args.target_columns,
                'task_type': args.task_type,
                'metric_type': args.metric_type,
                'split_type': args.split_type
                }

        endtoend(config_file, dataset_name,
                 outer_k=int(args.outer_folds), outer_processes=int(args.outer_processes),
                 inner_k=int(args.inner_folds), inner_processes=int(args.inner_processes),
                 output_dir=args.result_folder, debug=args.debug, data_stats=data_dict)
    
