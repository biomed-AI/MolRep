# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica et al "A Fair Comparison of Graph Neural Networks for Graph classification" -> https://github.com/diningphil/gnn-comparison
"""


import torch
from torch_geometric import data
from torch_geometric.data import Data
from typing import List, Optional

from molrep.data.datasets.base_dataset import MoleculeDataset
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

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.y for d in self._data]

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            target = torch.FloatTensor([targets[i]])
            self._data[i].y = target.unsqueeze(1) if target.ndim == 1 else target


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
