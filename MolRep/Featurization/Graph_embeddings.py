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

import torch
import rdkit
from rdkit import Chem
from pathlib import Path

from networkx import normalized_laplacian_matrix

from MolRep.Featurization.utils.graph_utils import *
from MolRep.Utils.utils import *


class GraphEmbeddings():
    def __init__(self, dataset_path, features_dir=None, use_node_attrs=True,
                 use_node_degree=False, use_one=False, max_reductions=10, precompute_kron_indices=True,
                 save_temp_data=True, configs=None, dataset_configs=None, logger=None):
        """
        Args:
            - dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                                the first one contains SMILES strings of the compounds,
                                the second one contains labels.
            - features_dir (str): A path to save processed features.
            - use_node_attrs (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
            - use_node_degree (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
            - use_one (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                    is present, the features will be saved after calculations. Defaults to True.
            - max_reductions (int): max reductions for KRON_REDUCTIONS.
            - precompute_kron_indices (bool): whether to precompute kron indices.
            - save_temp_data (bool): whether to save txt file when generating features.
            - configs (Namespace): Namespace of basic configuration.
            - dataset_configs (dict): Namespace of dataset configuration.
            - logger (logging): logging.
        """

        self.model_name = configs.model_name
        self.dataset_name = configs.dataset_name
        self.dataset_path = Path(dataset_path) / Path(dataset_configs["path"])
        self.features_dir = Path(dataset_path).parent if features_dir is None else Path(features_dir)
        self.temp_dir = self.features_dir / "raw" / self.dataset_name
        self.configs = configs

        self.use_node_degree = use_node_degree
        self.use_node_attrs = use_node_attrs
        self.use_one = use_one

        self.task_type = dataset_configs["task_type"]
        self.multi_class = self.task_type == 'Multiclass-Classification'
        self.multiclass_num_classes = dataset_configs["multiclass_num_classes"] if self.task_type == 'Multi-Classification' else None

        self.smiles_col = dataset_configs["smiles_column"]
        self.target_cols = dataset_configs["target_columns"]
        self.num_tasks = len(self.target_cols)

        self.KRON_REDUCTIONS = max_reductions
        self.precompute_kron_indices = precompute_kron_indices
        self.save_temp_data = save_temp_data
        self.logger = logger

        self.output_dir = self.features_dir #/ f"processed" / f"{self.dataset_name}"
        create_dir_if_not_exists(self.output_dir)
        create_dir_if_not_exists(self.temp_dir)

        if self.dataset_path.suffix == '.csv':
            self.whole_data_df = pd.read_csv(self.dataset_path)
        elif self.dataset_path.suffix == '.sdf':
            self.whole_data_df = self.load_sdf_files(self.dataset_path)
        else: 
            raise self.logger.error(f"File Format must be in ['CSV', 'SDF']")

        valid_smiles = filter_invalid_smiles(list(self.whole_data_df.loc[:,self.smiles_col]))
        self.whole_data_df = self.whole_data_df[self.whole_data_df[self.smiles_col].isin(valid_smiles)].reset_index(drop=True)

        self.configs.output_size = self.num_tasks * self.multiclass_num_classes if self.multi_class else self.num_tasks
        self.configs.train_data_size = len(self.whole_data_df)

    def load_sdf_files(self, input_file, clean_mols=True):
        suppl = Chem.SDMolSupplier(str(input_file), clean_mols, False, False)

        df_rows = []
        for ind, mol in enumerate(suppl):
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol)
            df_row = [ind+1, smiles, mol]
            df_rows.append(df_row)
        mol_df = pd.DataFrame(df_rows, columns=('mol_id', 'smiles', 'mol')).set_index('mol_id')
        try:
            raw_df = pd.read_csv(str(input_file) + '.csv').set_index('gdb9_index')
        except KeyError:
            raw_df = pd.read_csv(str(input_file) + '.csv')
            new = raw_df.mol_id.str.split('_', n = 1, expand=True)
            raw_df['mol_id'] = new[1]
            raw_df.set_index('mol_id')
        return pd.concat([mol_df, raw_df], axis=1, join='inner').reset_index(drop=True)

    def load_data_from_df(self):
        temp_dir = self.temp_dir

        df = self.whole_data_df
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
                mol = Chem.MolFromSmiles(smiles)
                num_nodes = len(mol.GetAtoms())
                node_dict = {}
                for i, atom in enumerate(mol.GetAtoms()):
                    node_dict[atom.GetIdx()] = cnt + i
                    fp_node_labels.writelines(str(atom.GetAtomicNum())+"\n")
                    fp_node_attrs.writelines(str(atom_features(atom))[1:-1]+"\n")

                fp_graph_indicator.write(f"{idx+1}\n"*num_nodes)  # node_i to graph id
                fp_graph_labels.write(','.join(['None' if np.isnan(g_label) else str(g_label) for g_label in g_labels])+"\n")
                for bond in mol.GetBonds():
                    node_1 = node_dict[bond.GetBeginAtomIdx()]
                    node_2 = node_dict[bond.GetEndAtomIdx()]
                    fp_edge_index.write(f"{node_1}, {node_2}\n{node_2}, {node_1}\n")
                    fp_edge_attrs.write(
                        str([1 if i else 0 for i in bond_features(bond)])[1:-1]+"\n")
                    fp_edge_attrs.write(
                        str([1 if i else 0 for i in bond_features(bond)])[1:-1]+"\n")
                cnt += num_nodes
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a Chem.Molecule.\nREASON: {}'.format(smiles, e))

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

        output_path = self.output_dir / f"{self.model_name}.pt"
        if os.path.exists(output_path):
            self.logger.info(f"Processed features existed.")
            self.logger.info(f"Loading features stored at '{output_path}'")

            dataset = torch.load(output_path)
            self.configs.dim_features = dataset[0].x.size(1)

            self.load_data_from_df()
            graphs_data, num_node_labels, num_edge_labels = parse_tu_data(self.model_name, self.temp_dir)

            # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
            self.configs.max_num_nodes = max([len(v) for (k, v) in graphs_data['graph_nodes'].items()])

        else:
            self.load_data_from_df()
            graphs_data, num_node_labels, num_edge_labels = parse_tu_data(self.model_name, self.temp_dir)
            targets = graphs_data.pop("graph_labels")

            # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
            self.configs.max_num_nodes = max([len(v) for (k, v) in graphs_data['graph_nodes'].items()])

            dataset = []
            for i, target in enumerate(targets, 1):
                graph_data = {k: v[i] for (k, v) in graphs_data.items()}
                G = create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels)

                if self.precompute_kron_indices:
                    laplacians, v_plus_list = self._precompute_kron_indices(G)
                    G.laplacians = laplacians
                    G.v_plus = v_plus_list

                data = self._to_data(G)
                dataset.append(data)

            self.configs.dim_features = dataset[0].x.size(1)
            torch.save(dataset, output_path)
            self.logger.info(f"Saving features at '{output_path}'")

        if self.save_temp_data == False:
            del_file(self.temp_dir)


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
        datadict.update(y=target)

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
