# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica et al "A Fair Comparison of Graph Neural Networks for Graph Classification" -> https://github.com/diningphil/gnn-comparison
"""

import os
import pandas as pd
import numpy as np
import copy

import torch
import rdkit
from rdkit import Chem
from pathlib import Path
from rdkit.Chem import AllChem

from networkx import normalized_laplacian_matrix
from ogb.graphproppred import PygGraphPropPredDataset

from MolRep.Featurization.utils.graph_utils import *
from MolRep.Utils.utils import *


class GraphEmbeddings():
    def __init__(self, data_df, model_name, features_path, dataset_config,
                 additional_data=None,
                 use_node_attrs=True, use_node_degree=False, use_one=False, max_reductions=10,
                 precompute_kron_indices=True,
                 save_temp_data=True):

        self.model_name = model_name
        self.whole_data_df = data_df
        self.features_path = features_path
        self.dataset_config = dataset_config
        self.dataset_name = self.dataset_config["name"]
        self.dataset_path = self.dataset_config["path"]
        self.additional_data = additional_data

        self.smiles_col = self.dataset_config["smiles_column"]
        self.target_cols = self.dataset_config["target_columns"]

        self.use_node_degree = use_node_degree
        self.use_node_attrs = use_node_attrs
        self.use_one = use_one

        self.temp_dir = self.features_path.parent / 'raw'
        if not self.temp_dir.exists():
            os.makedirs(self.temp_dir)

        self.KRON_REDUCTIONS = max_reductions
        self.precompute_kron_indices = precompute_kron_indices
        self.save_temp_data = save_temp_data


    @property
    def dim_features(self):
        return self._dim_features

    @property
    def dim_edge_features(self):
        return self._dim_edge_features

    @property
    def max_num_nodes(self):
        return self._max_num_nodes

    def load_data_from_df(self):
        temp_dir = self.temp_dir

        df = self.whole_data_df
        fp_smiles = open(temp_dir / f"{self.model_name}_SMILES.txt", "w")
        fp_edge_index = open(temp_dir / f"{self.model_name}_A.txt", "w")
        fp_graph_indicator = open(temp_dir / f"{self.model_name}_graph_indicator.txt", "w")
        fp_graph_labels = open(temp_dir / f"{self.model_name}_graph_labels.txt", "w")
        fp_node_labels = open(temp_dir / f"{self.model_name}_node_labels.txt", "w")
        fp_node_attrs = open(temp_dir / f"{self.model_name}_node_attributes.txt", "w")
        fp_edge_attrs = open(temp_dir / f"{self.model_name}_edge_attributes.txt", "w")
        
        cnt = 1
        for idx, row in df.iterrows():
            smiles, g_labels = row[self.smiles_col], row[self.target_cols].values
            try:
                fp_smiles.writelines(smiles+"\n")
                mol = Chem.MolFromSmiles(smiles)
                num_nodes = len(mol.GetAtoms())
                node_dict = {}
                for i, atom in enumerate(mol.GetAtoms()):
                    node_dict[atom.GetIdx()] = cnt + i
                    fp_node_labels.writelines(str(atom.GetAtomicNum()) + "\n")
                    fp_node_attrs.writelines(str(atom_features(atom))[1:-1] + "\n")

                fp_graph_indicator.write(f"{idx + 1}\n" * num_nodes)  # node_i to graph id
                fp_graph_labels.write(
                    ','.join(['None' if np.isnan(g_label) else str(g_label) for g_label in g_labels]) + "\n")

                for bond in mol.GetBonds():
                    node_1 = node_dict[bond.GetBeginAtomIdx()]
                    node_2 = node_dict[bond.GetEndAtomIdx()]
                    fp_edge_index.write(f"{node_1}, {node_2}\n{node_2}, {node_1}\n")
                    fp_edge_attrs.write(
                        str([1 if i else 0 for i in bond_features(bond)])[1:-1] + "\n")
                    fp_edge_attrs.write(
                        str([1 if i else 0 for i in bond_features(bond)])[1:-1] + "\n")
                cnt += num_nodes
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a Chem.Molecule.\nREASON: {}'.format(smiles, e))

        fp_smiles.close()
        fp_graph_labels.close()
        fp_node_labels.close()
        fp_graph_indicator.close()
        fp_edge_index.close()
        fp_node_attrs.close()
        fp_edge_attrs.close()

    def process(self):
        """
        Load and featurize data stored in a CSV file.
        """

        features_path = self.features_path

        if os.path.exists(features_path):
            dataset = torch.load(features_path)
            self._dim_features = dataset[0].x.size(1)
            self._dim_edge_features = dataset[0].edge_attr.size(1)

            # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
            self._max_num_nodes = dataset[0].max_num_nodes if hasattr(dataset[0], 'max_num_nodes') else max([da.x.shape[0] for da in dataset])

        elif self.dataset_name.startswith('ogb'):
            
            pyg_dataset = PygGraphPropPredDataset(name=self.dataset_name.replace('_', '-'), root=self.dataset_path)
            dataset = [data for data in pyg_dataset]
            
            self._dim_features = dataset[0].x.size(1)
            self._dim_edge_features = dataset[0].edge_attr.size(1)
            self._max_num_nodes = dataset[0].max_num_nodes if hasattr(dataset[0], 'max_num_nodes') else max([da.x.shape[0] for da in dataset])
            torch.save(dataset, features_path)
    
        else:
            self.load_data_from_df()
            graphs_data, num_node_labels, num_edge_labels = parse_tu_data(self.model_name, self.temp_dir)
            targets = graphs_data.pop("graph_labels")
            smiles_list = graphs_data.pop("smiles")

            # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
            self._max_num_nodes = max([len(v) for (k, v) in graphs_data['graph_nodes'].items()])

            dataset = []
            for i, (target, smiles) in enumerate(zip(targets, smiles_list), 1):
                graph_data = {k: v[i] for (k, v) in graphs_data.items()}
                G = create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels, smiles=smiles)

                G.max_num_nodes = self._max_num_nodes
                if self.precompute_kron_indices:
                    laplacians, v_plus_list = self._precompute_kron_indices(G)
                    G.laplacians = laplacians
                    G.v_plus = v_plus_list

                data = self._to_data(G)
                dataset.append(data)

            self._dim_features = dataset[0].x.size(1)
            self._dim_edge_features = dataset[0].edge_attr.size(1)
            torch.save(dataset, features_path)

        if self.save_temp_data == False:
            del_file(self.temp_dir)

        return dataset

    def _to_data(self, G):
        datadict = {}

        node_features = G.get_x(self.use_node_attrs,
                                self.use_node_degree, self.use_one)
        datadict.update(x=node_features)

        if G.laplacians is not None:
            datadict.update(laplacians=G.laplacians)
            datadict.update(v_plus=G.v_plus)

        edge_index = G.get_edge_index()
        datadict.update(edge_index=edge_index)

        if G.number_of_edges() and G.has_edge_attrs:
            edge_attr = G.get_edge_attr()
            datadict.update(edge_attr=edge_attr)
        else:
            edge_attr = torch.Tensor([])
            datadict.update(edge_attr=edge_attr)

        target = G.get_target()
        smiles = G.get_smiles()
        datadict.update(y=target)
        datadict.update(smiles=smiles)
        datadict.update(max_num_nodes=G.max_num_nodes)

        data = Data(**datadict)
        return data

    def _precompute_kron_indices(self, G):
        laplacians = []  # laplacian matrices (represented as 1D vectors)
        v_plus_list = []  # reduction matrices

        X = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        lap = torch.Tensor(normalized_laplacian_matrix(G).todense())  # I - D^{-1/2}AD^{-1/2}

        laplacians.append(lap)

        for _ in range(self.KRON_REDUCTIONS):
            if lap.shape[0] == 1:  # Can't reduce further:
                v_plus, lap = torch.tensor([1]), torch.eye(1)
            else:
                v_plus, lap = self._vertex_decimation(lap)

            laplacians.append(lap.clone())
            v_plus_list.append(v_plus.clone().long())

        return laplacians, v_plus_list

    # For the Perron–Frobenius theorem, if A is > 0 for all ij then the leading eigenvector is > 0
    # A Laplacian matrix is symmetric (=> diagonalizable)
    # and dominant eigenvalue (true in most cases? can we enforce it?)
    # => we have sufficient conditions for power method to converge
    def _power_iteration(self, A, num_simulations=30):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = torch.rand(A.shape[1]).unsqueeze(dim=1) * 0.5 - 1

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = torch.mm(A, b_k)

            # calculate the norm
            b_k1_norm = torch.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k

    def _vertex_decimation(self, L):

        max_eigenvec = self._power_iteration(L)
        v_plus, v_minus = (max_eigenvec >= 0).squeeze(
        ), (max_eigenvec < 0).squeeze()

        # diagonal matrix, swap v_minus with v_plus not to incur in errors (does not change the matrix)
        if torch.sum(v_plus) == 0.:  # The matrix is diagonal, cannot reduce further
            if torch.sum(v_minus) == 0.:
                assert v_minus.shape[0] == L.shape[0], (v_minus.shape, L.shape)
                # assumed v_minus should have ones, but this is not necessarily the case. So added this if
                return torch.ones(v_minus.shape), L
            else:
                return v_minus, L

        L_plus_plus = L[v_plus][:, v_plus]
        L_plus_minus = L[v_plus][:, v_minus]
        L_minus_minus = L[v_minus][:, v_minus]
        L_minus_plus = L[v_minus][:, v_plus]

        L_new = L_plus_plus - \
                torch.mm(torch.mm(L_plus_minus, torch.inverse(
                    L_minus_minus)), L_minus_plus)

        return v_plus, L_new
