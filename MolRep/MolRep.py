# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie
'''


import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from MolRep.Utils.utils import *
from MolRep.Utils.config_from_dict import Config, Grid, DatasetConfig
from MolRep.Utils.parser import get_basic_configs

from MolRep.Models import netWrapper
from MolRep.Models.losses import get_loss_func
from MolRep.Models.schedulers import build_lr_scheduler
from MolRep.Experiments.Graph_Data import Graph_data, MPNN_data
from MolRep.Experiments.Sequence_Data import Sequence_data, MAT_data
from MolRep.Experiments.Unsupervised_Data import Mol2Vec_data, NGramGraph_data, VAE_data

from MolRep.preprocess import prepare_features
from MolRep.data_split import DataSplit, k_fold_data_splits


class DataLoaderWrapper():
    def __init__(self, model_name, features_path, split_path=None, batch_size=32, shuffle=False, mode='training', data_size=None, configs=None, features_scaling=None, dataset_configs=None, logger=None):
        '''
        Args:
            - model_name (str): The name of Model.
            - features_path (str): A path to save processed features. (pt path)
            - split_path (str): A path to save data splits. (json)
            - batch_size (int): Batch size.
            - shuffle (bool): If True the data will be loaded in a random order. Defaults to False.
            - configs (Namespace): Namespace of basic configuration.
            - dataset_configs (dict): Namespace of dataset configuration.
            - logger (logging): logging.
        '''

        self.model_name = model_name
        self.batch_size = batch_size
        self.split_path = split_path
        self.features_path = features_path
        self.shuffle = shuffle

        self.configs = configs
        self.logger = logger

        self.features_scaling = features_scaling
        self.task_type = dataset_configs["task_type"]

        if mode == 'testing':
            self.splits = []
            self.splits.append({'test': np.arange(data_size)})
        else:
            self.splits = self.get_splits_indices()

    def get_splits_indices(self):
        with open(self.split_path, "r") as fp:
            splits = json.load(fp)
        return splits

    def get_loader_one_fold(self, split_idx: int = 0):
        indices = self.splits[split_idx]
        trainset_indices = indices['train'] if 'train' in indices.keys() else None
        validset_indices = indices['valid'] if 'valid' in indices.keys() else None
        testset_indices = indices['test']

        train_loader, valid_loader, test_loader = None, None, None

        if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool']:
            train_dataset, valid_dataset, test_dataset = Graph_data.Graph_construct_dataset(
                self.features_path, trainset_indices, valid_idxs=validset_indices, test_idxs=testset_indices)
            train_loader, valid_loader, test_loader, features_scaler, scaler = Graph_data.Graph_construct_dataloader(
                train_dataset, validset=valid_dataset, testset=test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, task_type=self.task_type, features_scaling=self.features_scaling)

        elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            train_dataset, valid_dataset, test_dataset = MPNN_data.MPNN_construct_dataset(
                self.features_path, trainset_indices, valid_idxs=validset_indices, test_idxs=testset_indices)
            train_loader, valid_loader, test_loader, features_scaler, scaler = MPNN_data.MPNN_construct_dataloader(
                train_dataset, validset=valid_dataset, testset=test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, task_type=self.task_type, features_scaling=self.features_scaling)

        elif self.model_name == 'MAT':
            train_dataset, valid_dataset, test_dataset = MAT_data.MAT_construct_dataset(
                    self.features_path, trainset_indices, valid_idxs=validset_indices, test_idxs=testset_indices)
            train_loader, valid_loader, test_loader, features_scaler, scaler = MAT_data.MAT_construct_loader(
                    train_dataset, validset=valid_dataset, testset=test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, task_type=self.task_type, features_scaling=self.features_scaling)

        elif self.model_name in ['BiLSTM', 'SALSTM', 'Transformer']:
            train_dataset, valid_dataset, test_dataset = Sequence_data.Sequence_construct_dataset(
                self.features_path, trainset_indices, valid_idxs=validset_indices, test_idxs=testset_indices)
            train_loader, valid_loader, test_loader, features_scaler, scaler = Sequence_data.Sequence_construct_loader(
                train_dataset, validset=valid_dataset, testset=test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, task_type=self.task_type, features_scaling=self.features_scaling)

        elif self.model_name == 'VAE':
            train_dataset, valid_dataset, test_dataset = VAE_data.VAE_construct_dataset(
                self.features_path, trainset_indices, valid_idxs=validset_indices, test_idxs=testset_indices)
            train_loader, valid_loader, test_loader, features_scaler, scaler = VAE_data.VAE_construct_loader(
                train_dataset, validset=valid_dataset, testset=test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, task_type=self.task_type, features_scaling=self.features_scaling)

        elif self.model_name in ['N_Gram_Graph', 'Mol2Vec']:
            if self.model_name == 'Mol2Vec':
                train_loader, valid_loader, test_loader, features_scaler, scaler = Mol2Vec_data.Mol2Vec_construct_loader(
                        self.features_path, trainset_indices, valid_idxs=validset_indices, test_idxs=testset_indices)
            elif self.model_name == 'N_Gram_Graph':
                train_loader, valid_loader, test_loader, features_scaler, scaler = NGramGraph_data.N_Gram_Graph_construct_loader(
                        self.features_path, trainset_indices, valid_idxs=validset_indices, test_idxs=testset_indices)

        else:
            raise self.logger.error(f"Model Name must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', \
                                               'MPNN', 'DMPNN', 'CMPNN', 'MAT', 'BiLSTM', 'BiLSTM-Attention']")

        return train_loader, valid_loader, test_loader, features_scaler, scaler


def construct_training_data(dataset_name = None,
                            model_name = None,
                            dataset_path = None,
                            smiles_colums = None,
                            target_columns = [],
                            task_type = None,
                            metric_type = None,
                            split_type = None,
                            k_fold = None,
                            split_sizes = [],
                            batch_size = 32,
                            split_dir = None,
                            features_dir = None,
                            shuffle = True,
                            seed = 0
                            ):

    assert (dataset_name is not None or dataset_path is not None), 'Dataset must be provided!'
    assert (k_fold is not None or sum(split_sizes) == 1)

    basic_configs = get_basic_configs()
    basic_configs.log_file = './logging.log'
    logger = create_logger(basic_configs)

    basic_configs.features_dir = Path(features_dir) / dataset_name if features_dir is not None else Path.cwd() / 'Data' / dataset_name
    basic_configs.split_dir = Path(split_dir) / dataset_name if features_dir is not None else Path.cwd() / 'Splits' / dataset_name

    create_dir_if_not_exists(basic_configs.features_dir)
    create_dir_if_not_exists(basic_configs.split_dir)

    if dataset_path is None and dataset_name in DatasetConfig.Data.keys():
        dataset_configs = DatasetConfig(dataset_name)
        dataset_path = Path(basic_configs.dataset_path) / dataset_configs["path"]
    else:
        dataset_configs = dict()
        dataset_configs["path"] = dataset_path
        dataset_configs["smiles_column"] = smiles_colums
        dataset_configs["target_columns"] = target_columns
        dataset_configs["task_type"] = task_type
        dataset_configs["metric_type"] = metric_type
        dataset_configs["split_type"] = split_type

    basic_configs.k_fold = k_fold
    basic_configs.model_name = model_name
    basic_configs.dataset_name = dataset_name
    basic_configs.split_sizes = split_sizes
    basic_configs.shuffle = shuffle
    basic_configs.seed = seed

    prepare_features(basic_configs, dataset_configs, logger)
    k_fold_data_splits(basic_configs.seed, basic_configs, dataset_configs=dataset_configs, logger=logger)

    features_path = basic_configs.features_dir / f"{basic_configs.model_name}.pt"
    split_path = basic_configs.split_dir / f"{basic_configs.dataset_name}_{dataset_configs['split_type']}_splits_seed{basic_configs.seed}.json"

    # Dataloader
    data_loader = DataLoaderWrapper(model_name, features_path, split_path,
                                    batch_size=batch_size, shuffle=shuffle,
                                    configs=basic_configs, dataset_configs=dataset_configs, logger=logger)

    if k_fold is not None:
        # print("K-fold Training. Return the list of K-fold dataloader: {'fold-0':(train_loader, test_loader), 'fold-1':(train_loader, test_loader), 'fold-2':(train_loader, test_loader)}")

        data = {}
        for fold_i in range(k_fold):
            train_loader, _, test_loader, features_scaler, scaler = data_loader.get_loader_one_fold(fold_i)
            data[str(fold_i)] = (train_loader, test_loader, features_scaler, scaler)
        return basic_configs, dataset_configs, data, logger

    elif k_fold is None and sum(split_sizes) == 1:
        # print("Train/Valid/Test Training. Return the list of [train_loader, valid_loader, test_loader]")

        train_loader, valid_loader, test_loader, features_scaler, scaler = data_loader.get_loader_one_fold()
        data = (train_loader, valid_loader, test_loader, features_scaler, scaler)
        return basic_configs, dataset_configs, data, logger

    else:
        raise "Data split must be K-fold or Train/Valid/Test. (parameter k-fold is None and sum of split_sizes is not equal 1)."


def construct_model(model_name,
                    model_config,
                    basic_configs,
                    dataset_configs,
                    parameter_idx = 0,
                    task_type = None,
                    multiclass_num_classes = None,
                    pretrained = False,
                    pretrained_model_path = None,
                    logger = None
                    ):

    basic_configs.model_config = Path(model_config)

    model_configurations = Grid(basic_configs.model_config)
    model_configuration = model_configurations[parameter_idx]
    model_configs = Config(**model_configuration)

    model_class = model_configs.model
    # Model
    model = model_class(basic_configs.dim_features, basic_configs.output_size, model_configs=model_configuration, dataset_configs=dataset_configs, configs=basic_configs)
    
    if pretrained:
        model = load_checkpoint(pretrained_model_path, model, basic_configs, logger)
    return model_configs, model


def construct_optimizer(model,
                        model_configs
                        ):

    optim_class = model_configs.optimizer
    # Optimizer
    optimizer = optim_class(model.parameters(),lr=model_configs['learning_rate'], weight_decay=model_configs['l2'])
    return optimizer

def construct_stopper(model_configs):
    return model_configs.early_stopper

def construct_clipping(model_configs):
    return model_configs.gradient_clipping

def construct_scheduler(optimizer,
                        model_configs,
                        basic_configs
                        ):

    scheduler = build_lr_scheduler(optimizer, model_configs=model_configs, configs=basic_configs)
    return scheduler

def construct_loss_func(dataset_configs,
                        basic_configs
                        ):

    loss_fn = get_loss_func(dataset_configs, basic_configs)
    return loss_fn

def construct_trainer(model,
                      basic_configs,
                      dataset_configs,
                      model_configs,
                      loss_fn):

    # Network for training
    net = netWrapper.NetWrapper(model, configs=basic_configs, dataset_configs=dataset_configs, model_config=model_configs, loss_function=loss_fn)
    return net

def training(net,
             train_loader,
             valid_loader = None,
             test_loader = None,
             optimizer = None,
             scheduler = None,
             clipping = None,
             stopper = None,
             scaler = None,
             target_idx = -1, 
             logger = None):

    metric, loss = net.train_test_one_fold(train_loader, valid_loader=valid_loader, test_loader=test_loader, optimizer=optimizer, 
                                           scheduler=scheduler, clipping=clipping, early_stopping=stopper,
                                           target_idx=target_idx, scaler=scaler, logger=logger)
    
    return net.model, metric, loss

def save_model(model_path,
               model, 
               scaler,
               features_scaler):

    save_checkpoint(model_path, model, scaler, features_scaler)

def construct_predicting_data(dataset_name = None,
                              dataset_path = None,
                              smiles_colums = None,
                              task_type = None,
                              metric_type = None,
                              model_name = None,
                              batch_size = 32,
                              features_dir = None,
                              shuffle = True,
                              seed = 0):

    basic_configs = get_basic_configs()
    logger = create_logger(basic_configs)

    basic_configs.features_dir = Path(features_dir) / dataset_name if features_dir is not None else Path.cwd() / 'Data' / dataset_name
    create_dir_if_not_exists(basic_configs.features_dir)
    features_path = basic_configs.features_dir / f"{model_name}.pt"

    if dataset_path is None and dataset_name in DatasetConfig.Data.keys():
        dataset_configs = DatasetConfig(basic_configs.dataset_name)
        dataset_path = Path(dataset_path) / self.dataset_configs["path"]
    else:
        dataset_configs = dict()
        dataset_configs["path"] = dataset_path
        dataset_configs["smiles_column"] = smiles_colums
        dataset_configs["task_type"] = task_type
        dataset_configs["metric_type"] = metric_type

    basic_configs.model_name = model_name
    basic_configs.dataset_name = dataset_name
    basic_configs.features_dir = features_dir
    basic_configs.shuffle = shuffle
    basic_configs.seed = seed

    prepare_features(basic_configs, dataset_configs, logger)

    # Dataloader
    data_loader = DataLoaderWrapper(model_name, features_path, batch_size=batch_size,
                                    shuffle=shuffle, mode='testing', data_size=basic_configs.train_data_size,
                                    configs=basic_configs, dataset_configs=dataset_configs, logger=logger)

    _, _, test_loader, _, _ = data_loader.get_loader_one_fold()
    return basic_configs, dataset_configs, test_loader

def predict(net,
            test_loader,
            scaler):

    y_preds, metric, loss = net.test_on_epoch_end(test_loader, scaler)
    return y_preds, metric