
# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie



CUDA_VISIBLE_DEVICES=2 python Training_Experiments.py --config_file ../MolRep/Configs/config_CMPNN.yml \
                                                      --dataset_name ogbg-molbbbp \
                                                      --validation_size 0.1 \
                                                      --test_size 0.1 \
                                                      --no_parallel


'''

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import os, random, torch
import numpy as np


from MolRep.Utils.config_from_dict import Config, Grid, DatasetConfig

# Model Selection
from MolRep.Evaluations.model_selection.KFoldSelection import KFoldSelector, KFoldBayesianSelector
from MolRep.Evaluations.model_selection.HoldOutSelection import HoldOutSelector, HoldOutBayesianSelector

# Model Assessment
from MolRep.Evaluations.model_assessment.KFoldAssessment import KFoldAssessment
from MolRep.Evaluations.model_assessment.HoldOutAssessment import HoldOutAssessment

from MolRep.Evaluations.DatasetWrapper import DatasetWrapper
from MolRep.Experiments.experiments import EndToEndExperiment


def endtoend(config_file, dataset_name,
             outer_k, outer_processes, inner_k, inner_processes,
             output_dir, datasets_dir=None, test_size=0., validation_size=0.,
             no_parallel=False, data_stats=None,
             hyper_search_type='Bayesian'):

    # Needed to avoid thread spawning, conflicts with multi-processing. You may set a number > 1 but take into account
    # the number of processes on the machine
    torch.set_num_threads(1)

    data_dir = Path(datasets_dir).parent / f"processed_data"
    split_dir = Path(datasets_dir).parent / f"splits"

    outer_k = int(outer_k) if outer_k is not None else None
    inner_k = int(inner_k) if inner_k is not None else None

    experiment_class = EndToEndExperiment

    model_configurations = Grid(config_file)
    model_configuration = Config(**model_configurations[0])
    dataset_configuration = DatasetConfig(dataset_name, data_dict=data_stats)
    if datasets_dir is not None:
        dataset_configuration.set_dataset_path(os.path.join(datasets_dir, dataset_configuration['path']))

    exp_path = os.path.join(output_dir, f'{model_configuration.exp_name}_{dataset_configuration.exp_name}_assessment')

    dataset = DatasetWrapper(dataset_config=dataset_configuration,
                             model_name=model_configuration.exp_name,
                             split_dir=split_dir, features_dir=data_dir,
                             outer_k=outer_k, inner_k=inner_k, 
                             test_size=test_size, validation_size=validation_size)
    
    if inner_k is not None:
        if hyper_search_type == "Bayesian":
            model_selector = KFoldBayesianSelector(folds=inner_k, max_processes=inner_processes)
        else:
            model_selector = KFoldSelector(folds=inner_k, max_processes=inner_processes)
    elif validation_size > 0:
        if hyper_search_type == "Bayesian":
            model_selector = HoldOutBayesianSelector(max_processes=inner_processes)
        else:
            model_selector = HoldOutSelector(max_processes=inner_processes)
    else:
        raise print('inner_k is None and validation_size is 0.')

    
    if outer_k is not None:
        risk_assesser = KFoldAssessment(outer_k, model_selector, exp_path, model_configurations, dataset_configuration,
                                        outer_processes=outer_processes)
    elif test_size > 0 or data_stats['split_type'] == 'defined':
        risk_assesser = HoldOutAssessment(model_selector, exp_path, model_configurations, dataset_configuration,
                                          max_processes=outer_processes)
    else:
        raise print("outer-k is None and test-size is 0 and split_type != 'defined'. ")

    risk_assesser.risk_assessment(dataset, experiment_class, no_parallel=no_parallel)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Following settings will reduced performance    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

if __name__ == "__main__":

    import argparse
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config_file')
    parser.add_argument('--experiment', dest='experiment', default='endtoend')
    parser.add_argument('--result_folder', dest='result_folder', default='../Outputs_Bayes/')
    parser.add_argument('--dataset_name', dest='dataset_name', default='none')
    parser.add_argument('--datasets_dir', dest='datasets_dir', default='../dataset/')
    parser.add_argument('--outer_k', dest='outer_k', default=None)
    parser.add_argument('--outer_processes', dest='outer_processes', type=int, default=3)
    parser.add_argument('--inner_k', dest='inner_k', default=None)
    parser.add_argument('--inner_processes', dest='inner_processes', type=int, default=2)
    parser.add_argument('--test_size', dest='test_size', type=float, default=0.)
    parser.add_argument('--validation_size', dest='validation_size', type=float, default=0.)
    parser.add_argument('--no_parallel', action="store_true", dest='no_parallel')

    parser.add_argument('--dataset_path', dest='dataset_path', default=None)
    parser.add_argument('--multiclass_num_classes', dest='multi-class num classes', type=int, default=1)
    parser.add_argument('--smiles_column', dest='smiles_column', default=None)
    parser.add_argument('--target_columns', dest='target_columns', default=[])
    parser.add_argument('--task_type', dest='task_type', default=None)
    parser.add_argument('--metric_type', dest='metric_type', default=None)
    parser.add_argument('--split_type', dest='split_type', default=None)
    parser.add_argument('--split_column', dest='split column', default=None)

    args = parser.parse_args()
    seed_everything(2022)

    config_file = args.config_file
    dataset_name = args.dataset_name
    experiment = args.experiment

    assert args.inner_k is not None or args.validation_size > 0 
    assert args.outer_k is not None or args.split_type == 'defined' or args.test_size > 0

    if args.dataset_name in ['QM7b', 'QM8', 'QM9', 'ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', \
                                    'HIV', 'PDBbind', 'BACE', 'BBBP', 'Tox21', 'SIDER', 'ClinTox'] or args.dataset_name.startswith('ogb'):

        endtoend(config_file, dataset_name,
                 datasets_dir=args.datasets_dir,
                 outer_k=args.outer_k, outer_processes=args.outer_processes,
                 inner_k=args.inner_k, inner_processes=args.inner_processes,
                 test_size=args.test_size, validation_size=args.validation_size,
                 output_dir=args.result_folder, no_parallel=args.no_parallel)
    
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
                 outer_k=args.outer_folds, outer_processes=args.outer_processes,
                 inner_k=args.inner_folds, inner_processes=args.inner_processes,
                 test_size=args.test_size, validation_size=args.validation_size,
                 output_dir=args.result_folder, no_parallel=args.no_parallel, data_stats=data_dict)
    
