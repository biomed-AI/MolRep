
# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie


CUDA_VISIBLE_DEVICES=1 python Training_Experiments.py --config_file ../MolRep/Configs/config_CMPNN.yml \
                                                      --dataset_name Metabolism
                                                      --dataset_path ../MolRep/Datasets/Metabolism/admet_exp_hlm_t1-2_20210412_TriCLF.csv \
                                                      --smiles_column COMPOUND_SMILES \
                                                      --target_columns CLF_LABEL \
                                                      --task_type Multi-Classification \
                                                      --multiclass_num_classes 3 \
                                                      --metric_type acc
                                                      --result_folder ../Outputs
                                                      --split_column SPLIT
                                                      --validation_size 0.2
'''

import os
import torch
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from MolRep.EndtoEnd_Experiments import endtoend


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', dest='config_file')
    parser.add_argument('--experiment', dest='experiment', default='endtoend')
    parser.add_argument('--result_folder', dest='result_folder', default='../Outputs/')
    parser.add_argument('--dataset_name', dest='dataset_name', default='none')
    parser.add_argument('--outer_k', dest='outer_k', default=None)
    parser.add_argument('--outer_processes', dest='outer_processes', type=int, default=3)
    parser.add_argument('--inner_k', dest='inner_k', default=None)
    parser.add_argument('--inner_processes', dest='inner_processes', type=int, default=2)
    parser.add_argument('--test_size', dest='test_size', type=float, default=0.)
    parser.add_argument('--validation_size', dest='validation_size', type=float, default=0.)
    parser.add_argument('--debug', action="store_true", dest='debug')

    parser.add_argument('--dataset_path', dest='dataset_path', default=None)
    parser.add_argument('--multiclass_num_classes', dest='multi-class num classes', type=int, default=1)
    parser.add_argument('--smiles_column', dest='smiles_column', default=None)
    parser.add_argument('--target_columns', dest='target_columns', default=[])
    parser.add_argument('--task_type', dest='task_type', default=None)
    parser.add_argument('--metric_type', dest='metric_type', default=None)
    parser.add_argument('--split_type', dest='split_type', default=None)
    parser.add_argument('--split_column', dest='split column', default=None)

    args = parser.parse_args()

    config_file = args.config_file
    dataset_name = args.dataset_name
    experiment = args.experiment

    assert args.inner_k is not None or args.validation_size > 0
    assert args.outer_k is not None or args.test_size > 0 or args.split_type == 'defined'

    if args.dataset_name in ['QM7b', 'QM8', 'QM9', 'ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', \
                                    'HIV', 'PDBbind', 'BACE', 'BBBP', 'Tox21', 'SIDER', 'ClinTox']:

        endtoend(config_file, dataset_name,
                outer_k=args.outer_folds, outer_processes=args.outer_processes,
                inner_k=args.inner_folds, inner_processes=args.inner_processes,
                test_size=args.test_size, validation_size=args.validation_size,
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
                 outer_k=args.outer_folds, outer_processes=args.outer_processes,
                 inner_k=args.inner_folds, inner_processes=args.inner_processes,
                 test_size=args.test_size, validation_size=args.validation_size,
                 output_dir=args.result_folder, debug=args.debug, data_stats=data_dict)
    
