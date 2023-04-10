# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Yang et al "Analyzing Learned Molecular Representations for Property Prediction" & "A Deep Learning Approach to Antibiotic Discovery" -> https://github.com/chemprop/chemprop
Song et al "Communicative Representation Learning on Attributed Molecular Graphs" -> https://github.com/SY575/CMPNN
"""

import os

from rdkit import Chem
from pathlib import Path

import numpy as np
import torch

from molrep.processors.features import get_features_generator
from molrep.common.registry import registry


@registry.register_processor("mpnn")
class MPNNEmbeddings:
    # def __init__(self, data_df, model_name, features_path, dataset_config,
    #              additional_data=None, features_generator=None, use_data_saving=True, atom_descriptors=None):
    def __init__(self, cfg, data_df, additional_data=None, features_generator=None, atom_descriptors=None):

        self.whole_data_df = data_df
        self.model_name = cfg.model_cfg.name
        self.dataset_config = cfg.datasets_cfg
        self.dataset_name = self.dataset_config["name"]
        self.additional_data = additional_data

        self.features_dir = Path(registry.get_path("features_root")) / self.dataset_name
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.features_dir / (self.model_name + ".pt")

        self.features_generator = features_generator
        self.use_data_saving = cfg.run_cfg.get("use_data_saving", True)
        self.atom_descriptors = atom_descriptors

        self.smiles_col = self.dataset_config["smiles_column"]
        self.target_cols = self.dataset_config["target_columns"]

    @property
    def dim_features(self):
        return self._dim_features

    @property
    def dim_edge_features(self):
        return None

    @property
    def max_num_nodes(self):
        return 200

    def process(self):
        """
        Load and featurize data stored in a CSV file.
        """

        features_path = self.features_path
        if self.use_data_saving and os.path.exists(features_path):
            # smiles_all, x_all, y_all = pickle.load(open(features_path, 'rb'))
            dataset = torch.load(features_path)
            self._dim_features = len(dataset["x_all"][0][0]) if dataset["x_all"][0][0] is not None else 0

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            smiles_all, x_all, y_all = self.load_data_from_smiles(data_x, data_y)

            self._dim_features = len(x_all[0][0]) if x_all[0][0] is not None else 0
            dataset = {
                "x_all": x_all,
                "y_all": y_all,
                "smiles_all": smiles_all,
            }
            # pickle.dump((smiles_all, x_all, y_all), open(features_path, "wb"))
            torch.save(dataset, features_path)

        return dataset


    def load_data_from_smiles(self, x_smiles, labels):
        """
         Load and featurize data from lists of SMILES strings and labels.
        Args:
            x_smiles (list[str]): A list of SMILES strings.
            labels (list[float]): A list of the corresponding labels.
        Returns:
            A tuple (SMILES, X, y) in which SMILES is a list of smiles string, X is a list of SMILES features,
            and y is a list of the corresponding labels.
        """
        smiles_all, x_all, y_all = [], [], []
        for smiles, label in zip(x_smiles, labels):
            try:
                mol = Chem.MolFromSmiles(smiles)
                features = []
                atom_features, atom_descriptors = None, None
                if self.features_generator is not None:
                    for fg in self.features_generator:
                        features_generator = get_features_generator(fg)
                        if mol is not None and mol.GetNumHeavyAtoms() > 0:
                            features.extend(features_generator(mol))
                    features = np.array(features)
                
                if len(features) == 0:
                    features = None

                # Fix nans in features
                if features is not None:
                    replace_token = 0
                    features = np.where(np.isnan(features), replace_token, features)

                if self.atom_descriptors == 'feature':
                    atom_features = self.get_atom_features(smiles)
                elif self.atom_descriptors == 'descriptor':
                    atom_descriptors = self.get_atom_descriptors(smiles)

                smiles_all.append(smiles)
                x_all.append([features, atom_features, atom_descriptors])
                if isinstance(label, np.ndarray):
                    y_all.append(label)
                else:
                    y_all.append([label])
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a Molecule in RDkit.\nREASON: {}'.format(smiles, e))

        return smiles_all, x_all, np.array(y_all)
