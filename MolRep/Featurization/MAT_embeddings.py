# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
Maziarka, Łukasz, et al. "Molecule Attention Transformer."  -> https://github.com/ardigen/MAT
"""


import os
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
import torch


class MATEmbeddings():
    def __init__(self, data_df, model_name, features_path, dataset_config,
                 add_dummy_node=True, one_hot_formal_charge=False, use_data_saving=True,):

        self.model_name = model_name
        self.whole_data_df = data_df
        self.features_path = features_path
        self.dataset_config = dataset_config

        self.add_dummy_node = add_dummy_node
        self.one_hot_formal_charge = one_hot_formal_charge
        self.use_data_saving = use_data_saving

        self.smiles_col = self.dataset_config["smiles_column"]
        self.target_cols = self.dataset_config["target_columns"]

    @property
    def dim_features(self):
        return self._dim_features

    @property
    def max_num_nodes(self):
        return None

    def process(self):
        """
        Load and featurize data stored in a CSV file.
        """

        features_path = self.features_path
        if self.use_data_saving and os.path.exists(features_path):
            # x_all, y_all = pickle.load(open(features_path, "rb"))
            dataset = torch.load(features_path)
            self._dim_features = dataset["x_all"][0][0].shape[1]

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            x_all, y_all = self.load_data_from_smiles(data_x, data_y, add_dummy_node=self.add_dummy_node,
                                                      one_hot_formal_charge=self.one_hot_formal_charge)

            self._dim_features = x_all[0][0].shape[1]
            if self.use_data_saving and not os.path.exists(features_path):
                dataset = {
                    "x_all": x_all,
                    "y_all": y_all,
                }
                # pickle.dump((x_all, y_all), open(features_path, "wb"))
                torch.save(dataset, features_path)

        return dataset

    def load_data_from_smiles(self, x_smiles, labels, add_dummy_node=True, one_hot_formal_charge=False):
        """
         Load and featurize data from lists of SMILES strings and labels.
        Args:
            - x_smiles (list[str]): A list of SMILES strings.
            - labels (list[float]): A list of the corresponding labels.
            - add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
            - one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

        Returns:
            A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
            and y is a list of the corresponding labels.
        """
        x_all, y_all = [], []
        for smiles, label in zip(x_smiles, labels):
            try:
                mol = MolFromSmiles(smiles)
                try:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, maxAttempts=5000)
                    AllChem.UFFOptimizeMolecule(mol)
                    mol = Chem.RemoveHs(mol)
                except:
                    AllChem.Compute2DCoords(mol)
                
                afm, adj, dist = self.featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
                x_all.append([afm, adj, dist])
                y_all.append(label)
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

        return x_all, np.array(y_all)

    def featurize_mol(self, mol, add_dummy_node, one_hot_formal_charge):
        """Featurize molecule.
        Args:
            - mol (rdchem.Mol): An RDKit Mol object.
            - add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
            - one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

        Returns:
            A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
        """
        node_features = np.array([self.get_atom_features(atom, one_hot_formal_charge)
                                  for atom in mol.GetAtoms()])

        adj_matrix = np.eye(mol.GetNumAtoms())
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom().GetIdx()
            end_atom = bond.GetEndAtom().GetIdx()
            adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

        conf = mol.GetConformer()
        pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                            for k in range(mol.GetNumAtoms())])
        dist_matrix = pairwise_distances(pos_matrix)

        if add_dummy_node:
            m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
            m[1:, 1:] = node_features
            m[0, 0] = 1.
            node_features = m

            m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
            m[1:, 1:] = adj_matrix
            adj_matrix = m

            m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
            m[1:, 1:] = dist_matrix
            dist_matrix = m

        return node_features, adj_matrix, dist_matrix

    def get_atom_features(self, atom, one_hot_formal_charge=True):
        """Calculate atom features.
        Args:
            - atom (rdchem.Atom): An RDKit Atom object.
            - one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

        Returns:
            A 1-dimensional array (ndarray) of atom features.
        """
        attributes = []

        attributes += self.one_hot_vector(
            atom.GetAtomicNum(),
            # [5, 6, 7, 8, 9, 15, 16, 17, 20, 35, 53, 999]
            list(np.arange(1000))
        )

        attributes += self.one_hot_vector(
            len(atom.GetNeighbors()),
            [0, 1, 2, 3, 4, 5]
        )

        attributes += self.one_hot_vector(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4]
        )

        if one_hot_formal_charge:
            attributes += self.one_hot_vector(
                atom.GetFormalCharge(),
                [-1, 0, 1]
            )
        else:
            attributes.append(atom.GetFormalCharge())

        attributes.append(atom.IsInRing())
        attributes.append(atom.GetIsAromatic())

        return np.array(attributes, dtype=np.float32)


    def one_hot_vector(self, val, lst):
        """Converts a value to a one-hot vector based on options in lst"""
        if val not in lst:
            val = lst[-1]
        return map(lambda x: x == val, lst)
