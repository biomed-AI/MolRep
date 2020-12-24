# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Youyang Deng, Weiming Li, Hui Yang, Jiancong Xie

Code based on: 
Zheng, Shuangjia, et al. "Identifying structure–property relationships through SMILES syntax analysis with self-attention mechanism." Journal of chemical information and modeling 59.2 (2019): 914-923.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from MolRep.Models.scalers import StandardScaler

class MoleculeDataset(Dataset):
    def __init__(self, lines, properties):
        self.lines = lines
        self.properties = properties

    def __getitem__(self, index):
        return self.lines[index], self.properties[index]

    def __len__(self):
        return len(self.properties)

    def targets(self):
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [p for p in self.properties]

    def normalize_features(self, scaler=None, replace_nan_token=0):
        return None

    def set_targets(self, targets):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.
        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        assert len(self.properties) == len(targets)
        for i in range(len(self.properties)):
            self.properties[i] = targets[i]


def VAE_construct_dataset(features_path, train_idxs=None, valid_idxs=None, test_idxs=None):
    x_all, y_all, mask, voc = pickle.load(open(features_path, "rb"))

    trainset = MoleculeDataset(np.array(x_all)[train_idxs], np.array(y_all)[train_idxs]) if train_idxs is not None else None
    validset = MoleculeDataset(np.array(x_all)[valid_idxs], np.array(y_all)[valid_idxs]) if valid_idxs is not None else None
    testset = MoleculeDataset(np.array(x_all)[test_idxs], np.array(y_all)[test_idxs]) if test_idxs is not None else None
    return trainset, validset, testset

def VAE_construct_loader(trainset=None, validset=None, testset=None, batch_size=1, shuffle=True, task_type='Classification', features_scaling=False):

    if features_scaling and trainset is not None:
        features_scaler = trainset.normalize_features(replace_nan_token=0)
        if validset is not None:
            validset.normalize_features(features_scaler)
        if testset is not None:
            testset.normalize_features(features_scaler)
    else:
        features_scaler = None

    if task_type == 'Regression' and trainset is not None:
        train_targets = trainset.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        trainset.set_targets(scaled_targets)
    else:
        scaler = None

    train_loader = DataLoader(trainset, batch_size, shuffle) if trainset is not None else None
    valid_loader = DataLoader(validset, batch_size, False) if validset is not None else None
    test_loader = DataLoader(testset, batch_size, False) if testset is not None else None
    return train_loader, valid_loader, test_loader, \
           features_scaler, scaler
