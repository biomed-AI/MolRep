
# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

--no_parallel
'''

import os
import torch
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


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
    parser.add_argument('--test_size', dest='test_size', default=0.1)
    parser.add_argument('--validation_size', dest='validation_size', default=0.1)
    parser.add_argument('--no_parallel', action="store_true", dest='no_parallel')

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
                                    'HIV', 'PDBbind', 'BACE', 'BBBP', 'Tox21', 'SIDER', 'ClinTox']:
        endtoend(config_file, dataset_name,
                outer_k=args.outer_folds, outer_processes=args.outer_processes,
                inner_k=args.inner_folds, inner_processes=args.inner_processes,
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
                 outer_k=int(args.outer_folds), outer_processes=int(args.outer_processes),
                 inner_k=int(args.inner_folds), inner_processes=int(args.inner_processes),
                 output_dir=args.result_folder, no_parallel=args.no_parallel, data_stats=data_dict)
    
