# -*- coding: utf-8 -*-
"""
Created on 2022.06.20

@author: Jiahua Rao

"""

import os
import json
import numpy as np
import pandas as pd

from rdkit import Chem

from pathlib import Path
from sklearn.model_selection import train_test_split
from itertools import repeat, product

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset
from MolRep.Interactions.utils import *

class InteractionDatasetWrapper:

    def __init__(self, dataset_config, model_name, seed=42, holdout_test_size=0.1, features_dir='Data'):

        self.dataset_config = dataset_config
        self.dataset_dir = self.dataset_config["path"]
        self.dataset_name = self.dataset_config["name"]
        self.split_type = self.dataset_config["split_type"]
        self.add_inverse_edge = self.dataset_config["add_inverse_edge"]
        self.additional_node_files = self.dataset_config["additional_node_files"]
        self.additional_edge_files = self.dataset_config["additional_edge_files"]
        
        self.model_name = model_name
        self.seed = seed
        self.holdout_test_size = holdout_test_size

        self.features_dir = Path(features_dir)
        self.features_path = self.features_dir / f"{self.dataset_name}" / f"{self.model_name}.pt"

        self.raw_dir = Path(self.dataset_dir) / "raw"

        self._max_num_nodes = None
        if not self.features_path.parent.exists():
            os.makedirs(self.features_path.parent)
        
        if self.dataset_name.startswith('ogbl'):
            self.data = PygLinkPropPredDataset(self.dataset_name.replace('_', '-'), root=self.features_dir)[0]
        elif not self.features_path.exists():
            self._load_raw_data()
            self.data, _ = torch.load(self.features_path)
        else:
            self.data, _ = torch.load(self.features_path)

        self.splits = self.get_edge_split()


    @property
    def num_nodes(self):
        if hasattr(self.data, 'num_nodes'):
            return self.data.num_nodes
        else:
            data = T.ToSparseTensor()(self.data)
            return data.adj_t.size(0)

    @property
    def num_node_feats(self):
        if hasattr(self.data, 'num_features'):
            if self.data.num_features:
                return self.data.num_features
            else:
                return 2235
        else:
            return 0


    @staticmethod
    def collate(data_list):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            data[key].append(item[key])
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(
                    item.__cat_dim__(key, item[key]))
            else:
                s = slices[key][-1] + 1
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            item = data_list[0][key]
            if torch.is_tensor(item) and len(data_list) > 1:
                data[key] = torch.cat(data[key],
                                      dim=data.__cat_dim__(key, item))
            elif torch.is_tensor(item):  # Don't duplicate attributes...
                data[key] = data[key][0]
            elif isinstance(item, int) or isinstance(item, float):
                data[key] = torch.tensor(data[key])

            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices


    def _load_raw_data(self, from_npz=False):

        if from_npz:
            # npz
            graph_list = read_raw_graph_from_npz(self.raw_dir, self.add_inverse_edge)
        else:
            # csv
            graph_list = read_raw_graph_from_csv(self.raw_dir, self.add_inverse_edge, additional_node_files = self.additional_node_files, additional_edge_files = self.additional_edge_files)
        
        pyg_graph_list = []
        
        for graph in tqdm(graph_list):
            g = Data()
            g.num_nodes = graph['num_nodes']
            g.edge_index = torch.from_numpy(graph['edge_index'])

            del graph['num_nodes']
            del graph['edge_index']

            if graph['edge_attr'] is not None:
                g.edge_attr = torch.from_numpy(graph['edge_attr'])
                del graph['edge_attr']

            if graph['node_feat'] is not None:
                g.x = torch.from_numpy(graph['node_feat'])
                del graph['node_feat']

            for key in self.additional_node_files:
                g[key] = torch.from_numpy(graph[key])
                del graph[key]

            for key in self.additional_edge_files:
                g[key] = torch.from_numpy(graph[key])
                del graph[key]

            pyg_graph_list.append(g)

        print('Saving...')
        torch.save(self.collate([pyg_graph_list]), self.features_path)

    
    def _load_raw_heterograph_data(self, from_npz=False):

        if from_npz:
            # npz
            graph_list = read_raw_heterograph_from_npz(self.raw_dir, self.add_inverse_edge)
        else:
            # csv
            graph_list = read_raw_heterograph_from_csv(self.raw_dir, self.add_inverse_edge, additional_node_files = self.additional_node_files, additional_edge_files = self.additional_edge_files)

        pyg_graph_list = []

        for graph in tqdm(graph_list):
            g = Data()
            
            g.__num_nodes__ = graph['num_nodes_dict']
            g.num_nodes_dict = graph['num_nodes_dict']

            # add edge connectivity
            g.edge_index_dict = {}
            for triplet, edge_index in graph['edge_index_dict'].items():
                g.edge_index_dict[triplet] = torch.from_numpy(edge_index)

            del graph['edge_index_dict']

            if graph['edge_feat_dict'] is not None:
                g.edge_attr_dict = {}
                for triplet in graph['edge_feat_dict'].keys():
                    g.edge_attr_dict[triplet] = torch.from_numpy(graph['edge_feat_dict'][triplet])

                del graph['edge_feat_dict']

            if graph['node_feat_dict'] is not None:
                g.x_dict = {}
                for nodetype in graph['node_feat_dict'].keys():
                    g.x_dict[nodetype] = torch.from_numpy(graph['node_feat_dict'][nodetype])

                del graph['node_feat_dict']

            for key in self.additional_node_files:
                g[key] = {}
                for nodetype in graph[key].keys():
                    g[key][nodetype] = torch.from_numpy(graph[key][nodetype])

                del graph[key]

            for key in self.additional_edge_files:
                g[key] = {}
                for triplet in graph[key].keys():
                    g[key][triplet] = torch.from_numpy(graph[key][triplet])

                del graph[key]

            pyg_graph_list.append(g)

        print('Saving...')
        torch.save(self.collate([pyg_graph_list]), self.features_path)
        return pyg_graph_list

        
    def get_edge_split(self):
        path = osp.join(self.dataset_dir, 'split', self.split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'train.pt')))
        valid = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'valid.pt')))
        test = replace_numpy_with_torchtensor(torch.load(osp.join(path, 'test.pt')))

        return {'train': train, 'valid': valid, 'test': test}