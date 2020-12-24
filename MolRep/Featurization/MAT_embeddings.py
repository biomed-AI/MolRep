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

from MolRep.Utils.utils import *

class MATEmbeddings():
    def __init__(self, dataset_path, features_dir=None, model_name=None,
                 add_dummy_node=True, one_hot_formal_charge=False, use_data_saving=True,
                  configs=None, dataset_configs=None, logger=None):
        """
        Args:
            - dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                                the first one contains SMILES strings of the compounds,
                                the second one contains labels.
            - features_dir (str): A path to save processed features.
            - add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
            - one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
            - use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                    is present, the features will be saved after calculations. Defaults to True.
        """

        self.dataset_name = configs.dataset_name
        self.model_name = configs.model_name
        self.dataset_path = Path(dataset_path) / Path(dataset_configs["path"])
        self.features_dir = Path(dataset_path).parent if features_dir is None else Path(features_dir)

        self.add_dummy_node = add_dummy_node
        self.one_hot_formal_charge = one_hot_formal_charge
        self.use_data_saving = use_data_saving

        self.task_type = dataset_configs["task_type"]
        self.multi_class = self.task_type == 'Multiclass-Classification'
        self.multiclass_num_classes = dataset_configs["multiclass_num_classes"] if self.multi_class else None
        self.configs = configs
        self.logger = logger

        self.smiles_col = dataset_configs["smiles_column"]
        self.target_cols = dataset_configs["target_columns"]
        self.num_tasks = len(self.target_cols)

        self.output_dir = self.features_dir # / f"processed" / f"{self.dataset_name}"
        create_dir_if_not_exists(self.output_dir)

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
            raw_df = pd.read_csv(str(input_file) + '.csv').set_index('mol_id')
        return pd.concat([mol_df, raw_df], axis=1, join='inner').reset_index(drop=True)

    def process(self):
        """
        Load and featurize data stored in a CSV file.
        """

        output_path = self.output_dir / f"{self.model_name}.pt"

        if self.use_data_saving and os.path.exists(output_path):
            self.logger.info(f"Processed features existed.")
            self.logger.info(f"Processed features stored at '{self.output_dir}'")

            x_all, y_all = pickle.load(open(output_path, "rb"))
            self.configs.dim_features = x_all[0][0].shape[1]

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            x_all, y_all = self.load_data_from_smiles(data_x, data_y, add_dummy_node=self.add_dummy_node,
                                                      one_hot_formal_charge=self.one_hot_formal_charge)

            self.configs.dim_features = x_all[0][0].shape[1]
            if self.use_data_saving and not os.path.exists(output_path):
                self.logger.info(f"Saving features at '{output_path}'")
                pickle.dump((x_all, y_all), open(output_path, "wb"))

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
            [5, 6, 7, 8, 9, 15, 16, 17, 20, 35, 53, 999]
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
