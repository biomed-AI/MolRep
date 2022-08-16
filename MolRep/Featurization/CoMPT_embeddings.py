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
import math

import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances

class CoMPTEmbeddings():
    def __init__(self, data_df, model_name, features_path, dataset_config, use_data_saving=True):

        self.model_name = model_name
        self.whole_data_df = data_df
        self.features_path = features_path
        self.dataset_config = dataset_config
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
            self._dim_features = dataset["x_all"][0][0].shape[1]  # x_all

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            x_all, y_all = self.load_data_from_smiles(data_x, data_y, atom_hidden=115, bond_hidden=13)   # revised!

            self._dim_features = x_all[0][0].shape[1]
            if self.use_data_saving and not os.path.exists(features_path):
                # pickle.dump((x_all, y_all), open(features_path, "wb"))
                dataset = {
                    "x_all": x_all,
                    "y_all": y_all,
                }
                torch.save(dataset, features_path)
        
        return dataset

    def load_data_from_smiles(self, x_smiles, labels, atom_hidden, bond_hidden):
        """
         Load and featurize data from lists of SMILES strings and labels.
        Args:
            - x_smiles (list[str]): A list of SMILES strings.
            - labels (list[float]): A list of the corresponding labels.

        Returns:
            A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
            and y is a list of the corresponding labels.
        """
        x_all, y_all = [], []
        for smiles, label in zip(x_smiles, labels):
            try:
                mol = MolFromSmiles(smiles)
                # Set Stereochemistry
                Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
                Chem.rdmolops.AssignStereochemistryFrom3D(mol)
                AllChem.ComputeGasteigerCharges(mol)
                afm, efm, adj = self.featurize_mol(mol, atom_hidden, bond_hidden)
                x_all.append([afm, efm, adj])
                y_all.append(label)
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

        return x_all, np.array(y_all)


    def featurize_mol(self, mol, atom_hidden, bond_hidden):
        """Featurize molecule.
        Args:
            - mol (rdchem.Mol): An RDKit Mol object.
            - add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
            - one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

        Returns:
            A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
        """
        node_features = np.array([self.get_atom_features(atom, atom_hidden) for atom in mol.GetAtoms()])

        # Get Bond features
        bond_features = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), bond_hidden))

        for bond in mol.GetBonds():
            begin_atom_idx = bond.GetBeginAtom().GetIdx()
            end_atom_idx = bond.GetEndAtom().GetIdx()
            bond_features[begin_atom_idx, end_atom_idx, :] = bond_features[end_atom_idx, begin_atom_idx, :] = self.get_bond_features(bond, bond_hidden)

        # Get Adjacency matrix without self loop
        adjacency_matrix = Chem.rdmolops.GetDistanceMatrix(mol).astype(np.float32)

        return node_features, bond_features, adjacency_matrix


    def get_atom_features(self, atom, atom_hidden):
        # 100+1=101 dimensions
        v1 = self.one_hot_vector(atom.GetAtomicNum(), [i for i in range(1, 101)])

        # 5+1=6 dimensions
        v2 = self.one_hot_vector(
            atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]
        )

        # 8 dimensions
        v4 = [
            atom.GetTotalNumHs(includeNeighbors=True) / 8,
            atom.GetDegree() / 4,
            atom.GetFormalCharge() / 8,
            atom.GetTotalValence() / 8,
            0 if math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(
                atom.GetDoubleProp('_GasteigerCharge')) else atom.GetDoubleProp('_GasteigerCharge'),
            0 if math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(
                atom.GetDoubleProp('_GasteigerHCharge')) else atom.GetDoubleProp('_GasteigerHCharge'),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing())
        ]

        # index for position encoding
        v5 = [
            atom.GetIdx() + 1  # start from 1
        ]

        attributes = np.concatenate([v1, v2, v4, v5], axis=0)

        # total for 32 dimensions
        assert len(attributes) == atom_hidden + 1
        return attributes


    def get_bond_features(self, bond, bond_hidden):
        # 4 dimensions
        v1 = self.one_hot_vector(
            bond.GetBondType(), [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
                ], add_unknown=False)

        # 6 dimensions
        v2 = self.one_hot_vector(
            bond.GetStereo(), [
                Chem.rdchem.BondStereo.STEREOANY,
                Chem.rdchem.BondStereo.STEREOCIS,
                Chem.rdchem.BondStereo.STEREOE,
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOTRANS,
                Chem.rdchem.BondStereo.STEREOZ], add_unknown=False)

        # 3 dimensions
        v4 = [
            int(bond.GetIsConjugated()),
            int(bond.GetIsAromatic()),
            int(bond.IsInRing())
        ]

        # total for 115+13=128 dimensions
        attributes = np.concatenate([v1, v2, v4])

        assert len(attributes) == bond_hidden
        return attributes

    def one_hot_vector(self, val, lst, add_unknown=True):
        """Converts a value to a one-hot vector based on options in lst"""
        if add_unknown:
            vec = np.zeros(len(lst) + 1)
        else:
            vec = np.zeros(len(lst))

        vec[lst.index(val) if val in lst else -1] = 1
        return vec
