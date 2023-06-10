# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica et al "A Fair Comparison of Graph Neural Networks for Graph classification" -> https://github.com/diningphil/gnn-comparison
"""

import os
import numpy as np

import torch
from pathlib import Path
from rdkit.Chem import AllChem

from molrep.common.registry import registry
from molrep.processors.features import mol_to_graph_data_obj


@registry.register_processor("graph")
class GraphEmbeddings:
    def __init__(self, cfg, data_df, additional_data=None):

        self.whole_data_df = data_df
        self.model_name = cfg.model_cfg.name
        self.dataset_config = cfg.datasets_cfg

        self.dataset_name = self.dataset_config["name"]
        self.additional_data = additional_data

        self.features_dir = Path(registry.get_path("features_root")) / self.dataset_name
        self.features_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset_config.feature == 'simple':
            print('using simple feature')
            self.features_path = self.features_dir / (self.model_name + "_simple.pt")
        else:
            self.features_path = self.features_dir / (self.model_name + ".pt")

        self.smiles_col = self.dataset_config["smiles_column"]
        self.target_cols = self.dataset_config["target_columns"]

    @property
    def dim_features(self):
        return self._dim_features

    @property
    def dim_edge_features(self):
        return self._dim_edge_features

    @property
    def max_num_nodes(self):
        return self._max_num_nodes

    @staticmethod
    def preprocess(smiles):
        if smiles is not None:
            if not isinstance(smiles, list):
                smiles = [smiles]
            rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles]
            dataset = []
            for i in range(len(smiles)):
                rdkit_mol = rdkit_mol_objs_list[i]
                if rdkit_mol is None:
                    continue
                data = mol_to_graph_data_obj(rdkit_mol)
                data.id = torch.tensor([i])
                data.smiles = smiles[i]
                dataset.append(data)
            return dataset

    def process(self):
        """
        Load and featurize data stored in a CSV file or the Input SMILES.
        """
        features_path = self.features_path

        if os.path.exists(features_path):
            dataset = torch.load(features_path)
            self._dim_features = dataset[0].x.size(1)
            self._dim_edge_features = dataset[0].edge_attr.size(1)

            # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
            self._max_num_nodes = dataset[0].max_num_nodes if hasattr(dataset[0], 'max_num_nodes') else max([da.x.shape[0] for da in dataset])

        else:
            smiles_list = self.whole_data_df[self.smiles_col]
            rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
            labels = self.whole_data_df[self.target_cols].values

            dataset = []
            if labels.ndim == 1:
                labels = np.expand_dims(labels, axis=1)

            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs_list[i]
                if rdkit_mol is None:
                    continue
                data = mol_to_graph_data_obj(rdkit_mol)
                data.id = torch.tensor([i])
                data.y = torch.tensor([labels[i]])
                data.smiles = smiles_list[i]

                if self.dataset_config.feature == 'simple':
                    # only retain the top two node/edge features
                    data.x = data.x[:,:2]
                    data.edge_attr = data.edge_attr[:,:2]

                dataset.append(data)

            self._dim_features = dataset[0].x.size(1)
            self._dim_edge_features = dataset[0].edge_attr.size(1)
            self._max_num_nodes = dataset[0].max_num_nodes if hasattr(dataset[0], 'max_num_nodes') else max([da.x.shape[0] for da in dataset])
            torch.save(dataset, features_path)

        return dataset
