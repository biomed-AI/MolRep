# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica et al "A Fair Comparison of Graph Neural Networks for Graph Classification" -> https://github.com/diningphil/gnn-comparison
"""

import threading
import numpy as np
import pandas as pd
import torch_geometric as pyg
from random import Random

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from torch_geometric import data
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from typing import Dict, Callable, List, Union, Optional, Iterator

import rdkit
from rdkit import Chem

from MolRep.Models.scalers import StandardScaler
from MolRep.Utils.utils import worker_init

class MoleculeDataset(Dataset):
    def __init__(self, data):
        self._data = data
        self._random = Random()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.y for d in self._data]

    def smiles(self) -> List[List[Optional[str]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.smiles for d in self._data]

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        return None

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.
        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.y for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)
        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.
        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])


class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


class Batch(data.Batch):
    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        laplacians = None
        v_plus = None

        if 'laplacians' in data_list[0]:
            laplacians = [d.laplacians[:] for d in data_list]
            v_plus = [d.v_plus[:] for d in data_list]

        copy_data = []
        for d in data_list:
            copy_data.append(Data(x=d.x.float(),
                                  y=d.y,
                                  edge_index=d.edge_index,
                                  edge_attr=d.edge_attr,
                                  v_outs=d.v_outs if hasattr(d, 'v_outs') else None,
                                  g_outs=d.g_outs if hasattr(d, 'g_outs') else None,
                                  e_outs=d.e_outs if hasattr(d, 'e_outs') else None,
                                  o_outs=d.o_outs if hasattr(d, 'o_outs') else None,
                                  smiles=d.smiles if hasattr(d, 'smiles') else None)
                             )

        batch = data.Batch.from_data_list(copy_data, follow_batch=follow_batch)
        batch['laplacians'] = laplacians
        batch['v_plus'] = v_plus

        return batch


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 2,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 follow_batch: list = ['x'],
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0

        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            # collate_fn=construct_molecule_batch,
            collate_fn=lambda data_list: Batch.from_data_list(
                                                    data_list, follow_batch),
            multiprocessing_context=self._context,
            timeout=self._timeout,
            worker_init_fn=worker_init,
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].y for index in self._sampler]

    @property
    def smiles(self):
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].smiles for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()



def _construct_dataset(dataset, indices):
    return MoleculeDataset([dataset[idx] for idx in indices])

def _construct_dataloader(dataset, batch_size, shuffle, seed=0, num_workers=0, class_balance=False):
    """Construct a data loader for the provided data.
    Args:
        data_set (): 
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.
    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    if dataset is not None:
        loader = MoleculeDataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    class_balance=class_balance,
                    shuffle=shuffle,
                    seed=seed,
                )
    else: 
        loader = None
    return loader

def Graph_construct_dataset(features_path, train_idxs=None, valid_idxs=None, test_idxs=None):
    dataset = torch.load(features_path)

    trainset = _construct_dataset(dataset, train_idxs) if train_idxs is not None else None
    validset = _construct_dataset(dataset, valid_idxs) if valid_idxs is not None else None
    testset = _construct_dataset(dataset, test_idxs) if test_idxs is not None else None
    return trainset, validset, testset

def Graph_construct_dataloader(trainset=None, validset=None, testset=None, batch_size=1, shuffle=True, seed=0, task_type='Classification', features_scaling=True):

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

    return _construct_dataloader(trainset, batch_size, shuffle, seed), \
           _construct_dataloader(validset, batch_size, False, seed), \
           _construct_dataloader(testset, batch_size, False, seed), \
           features_scaler, scaler