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
from torch.utils.data import DataLoader

from molrep.common.utils import worker_init
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
    
    @property
    def smiles(self) -> List[str]:
        return [d.smiles for d in self._data]

    @classmethod
    def construct_dataset(cls, indices, features_path):
        features = torch.load(features_path)
        return cls([features[idx] for idx in indices])

    def collate_fn(self, batch, **kwargs):
        follow_batch = kwargs["follow_batch"] if "follow_batch" in kwargs.keys() else []
        batch_data = Batch.from_data_list(batch, follow_batch)
        return {
            "pygdata": batch_data,
            "targets": batch_data.y,
        }

    # def bulid_dataloader(self, config=None, is_train=True, **kwargs):
    #     if config is not None:
    #         num_workers = config.run_cfg.get("num_workers", 0)
    #         class_balance = config.run_cfg.get("class_balance", False)
    #         follow_batch = config.run_cfg.get("follow_batch", [])

    #         seed = config.run_cfg.get("seed", 42)
    #         batch_size = config.run_cfg.get("batch_size", 50)

    #     else:
    #         num_workers = kwargs["num_workers"] if "num_workers" in kwargs.keys() else 0
    #         class_balance = kwargs["class_balance"] if "class_balance" in kwargs.keys() else False
    #         follow_batch = kwargs["follow_batch"] if "follow_batch" in kwargs.keys() else []

    #         seed = kwargs["seed"] if "seed" in kwargs.keys() else 42
    #         batch_size = kwargs["batch_size"] if "batch_size" in kwargs.keys() else 50

    #     shuffle = (is_train == True)
    #     self._context = None
    #     self._timeout = 0
    #     is_main_thread = threading.current_thread() is threading.main_thread()
    #     if not is_main_thread and num_workers > 0:
    #         self._context = 'forkserver'  # In order to prevent a hanging
    #         self._timeout = 3600  # Just for sure that the DataLoader won't hang

    #     self._sampler = MoleculeSampler(
    #         dataset=self._data,
    #         class_balance=class_balance,
    #         shuffle=shuffle,
    #         seed=seed
    #     )

    #     return DataLoader(
    #                     dataset=self._data,
    #                     batch_size=batch_size,
    #                     sampler=self._sampler,
    #                     num_workers=num_workers,
    #                     collate_fn=lambda data_list: self.collate_fn(data_list, follow_batch),
    #                     timeout=self._timeout,
    #                     worker_init_fn=worker_init,
    #     )

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
                                  edge_attr=d.edge_attr.float(),
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
