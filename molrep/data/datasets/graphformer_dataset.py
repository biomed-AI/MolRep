# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica et al "A Fair Comparison of Graph Neural Networks for Graph Classification" -> https://github.com/diningphil/gnn-comparison
"""


import torch
from torch_geometric import data
from typing import List, Optional

import numpy as np

import algos
from molrep.data.datasets.base_dataset import MoleculeDataset
from molrep.common.registry import registry


@registry.register_dataset("graphformer")
class GraphformerDataset(MoleculeDataset):
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
        item = self._data[index]
        item.idx = index
        return preprocess_item(item)

    @property
    def smiles(self) -> List[str]:
        return [d.smiles for d in self._data]

    @classmethod
    def construct_dataset(cls, indices, features_path):
        features = torch.load(features_path)
        return cls([features[idx] for idx in indices])

    def collate_fn(self, batch, **kwargs):
        max_node = kwargs["max_node"] if "max_node" in kwargs.keys() else 512
        multi_hop_max_dist = kwargs["multi_hop_max_dist"] if "multi_hop_max_dist" in kwargs.keys() else 20
        rel_pos_max = kwargs["rel_pos_max"] if "rel_pos_max" in kwargs.keys() else 20

        batch_data = collator(batch, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist, rel_pos_max=rel_pos_max)
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


class Batch():
    def __init__(self, idx, attn_bias, attn_edge_type, rel_pos, in_degree, out_degree, x, edge_input, y):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.rel_pos = attn_bias, attn_edge_type, rel_pos
        self.edge_input = edge_input

    def to(self, device, non_blocking=True):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = self.in_degree.to(
            device), self.out_degree.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.rel_pos = self.attn_bias.to(
            device), self.attn_edge_type.to(device), self.rel_pos.to(device)
        self.edge_input = self.edge_input.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def collator(items, max_node=512, multi_hop_max_dist=20, rel_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.rel_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y) for item in items]
    idxs, attn_biases, attn_edge_types, rel_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][rel_poses[idx] >= rel_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num)
                        for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in out_degrees])
    return Batch(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        rel_pos=rel_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=edge_input,
        y=y,
    )

def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    # max_dist = 1
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    # edge_input = algos.gen_edge_input(max_dist, adj.numpy(), attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    # rel_pos = torch.from_numpy((adj.numpy())).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    return item
