# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica et al "A Fair Comparison of Graph Neural Networks for Graph Classification" -> https://github.com/diningphil/gnn-comparison
"""


import threading

import torch
from torch_geometric import data
from torch_geometric.data import Data
from typing import List, Optional

from molrep.data.datasets.base_dataset import MoleculeDataset, MoleculeSampler
from molrep.common.registry import registry


@registry.register_dataset("graph")
class GraphDataset(MoleculeDataset):
    def __init__(self, data):
        r"""
        Args:
            data: A list of :class:`PyG Data`\ s.
        """
        super().__init__(data)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).
        return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, index):
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.
        Args:
            item: An index (int) or a slice object.
        return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[index]
    
    def construct_data(cls, indices):
        features = torch.load(cls.features_path)
        return cls([features[idx] for idx in indices])

    @classmethod
    def collate_fn(cls, batch, follow_batch=[]):
        batch_data = Batch.from_data_list(batch, follow_batch)
        return {
            "data": batch_data,
            "targets": batch_data.y,
        }

    def bulid_dataloader(self, is_train=True):

        num_workers = self.config.run_cfg.get("num_workers", 2)
        class_balance = self.config.run_cfg.get("class_balance", False)
        shuffle = (is_train == True)
        seed = self.config.run_cfg.get("seed", 42)

        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._data,
            class_balance=class_balance,
            shuffle=shuffle,
            seed=seed
        )

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.y for d in self._data]


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
