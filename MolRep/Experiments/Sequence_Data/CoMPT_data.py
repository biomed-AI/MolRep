"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
Maziarka, Łukasz, et al. "Molecule Attention Transformer."  -> https://github.com/ardigen/MAT
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd

import torch
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

from MolRep.Models.scalers import StandardScaler

class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x, y, index):
        self.node_features = x[0]
        self.edge_features = x[1]
        self.adjacency_matrix = x[2]
        self.y = y
        self.index = index
    
    def set_targets(self, target):
        self.y = target


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]

    def normalize_features(self, scaler = None, replace_nan_token = 0):
        return None

    def targets(self):
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.y for d in self.data_list]

    def set_targets(self, targets):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.
        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        assert len(self.data_list) == len(targets)
        for i in range(len(self.data_list)):
            self.data_list[i].set_targets(targets[i])


def pad_array(array, shape, dytpe=np.float32):
    """Pad a 2/3-dimensional array with zeros.
    Args:
        array (ndarray): A 2/3-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.
    Returns:
        A 2/3-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dytpe)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array


def mol_collate_func(batch):
    """Create a padded batch of molecule features.
    Args:
        batch (list[Molecule]): A batch of raw molecules.
    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """

    node_features_list, edge_features_list, adjacency_list = [], [], []
    labels = []
    max_size = 0
    for molecule in batch:
        labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        node_features_list.append(pad_array(molecule.node_features,(max_size, molecule.node_features.shape[-1])))
        edge_features_list.append(pad_array(molecule.edge_features,(max_size, max_size, molecule.edge_features.shape[-1])))
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))

    return [torch.tensor(features) for features in (node_features_list, edge_features_list, adjacency_list, labels)]


def _construct_dataset(x_all, y_all):
    """Construct a MolDataset object from the provided data.
    Args:
        x_all (list): A list of molecule features.
        y_all (list): A list of the corresponding labels.
    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[1], i) for i, data in enumerate(zip(x_all, y_all))]
    return MolDataset(output)


def _construct_loader(data_set, batch_size, shuffle=True):
    """Construct a data loader for the provided data.
    Args:
        data_set (): 
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.
    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    if data_set is not None:
        loader = torch.utils.data.DataLoader(dataset=data_set,
                                            batch_size=batch_size,
                                            collate_fn=mol_collate_func,
                                            shuffle=shuffle)
    else:
        loader = None
    return loader


def CoMPT_construct_dataset(features_path, train_idxs=None, valid_idxs=None, test_idxs=None):
    # x_all, y_all = pickle.load(open(features_path, "rb"))
    dataset = torch.load(features_path)
    x_all, y_all = dataset["x_all"], dataset["y_all"]

    trainset = _construct_dataset(np.array(x_all)[train_idxs], np.array(y_all)[train_idxs]) if train_idxs is not None else None
    validset = _construct_dataset(np.array(x_all)[valid_idxs], np.array(y_all)[valid_idxs]) if valid_idxs is not None else None
    testset = _construct_dataset(np.array(x_all)[test_idxs], np.array(y_all)[test_idxs]) if test_idxs is not None else None
    return trainset, validset, testset


def CoMPT_construct_loader(trainset=None, validset=None, testset=None, batch_size=1, shuffle=True, task_type='Classification', features_scaling=True):
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
    return _construct_loader(trainset, batch_size, shuffle), \
           _construct_loader(validset, batch_size, False), \
           _construct_loader(testset, batch_size, False), \
           features_scaler, scaler
