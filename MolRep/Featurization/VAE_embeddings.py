"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on: 
Zheng, Shuangjia, et al. "Identifying structure–property relationships through SMILES syntax analysis with self-attention mechanism." Journal of chemical information and modeling 59.2 (2019): 914-923.
"""


import os, re
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles

class VAEEmbeddings:
    def __init__(self, data_df, model_name, features_path, dataset_config,
                 use_data_saving=True):

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
            _, _, _, voc = pickle.load(open(features_path, 'rb'))
            self._dim_features = len(voc)

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            x_all, y_all, voc = self.load_data_from_smiles(data_x, data_y)

            seq_data, label, mask = self.make_variables(x_all, y_all, voc)
            self._dim_features = len(voc)

            pickle.dump((seq_data, label, mask, voc), open(features_path, "wb"))

    def load_data_from_smiles(self, x_smiles, labels):
        """
         Load and featurize data from lists of SMILES strings and labels.
        Args:
            - x_smiles (list[str]): A list of SMILES strings.
            - labels (list[float]): A list of the corresponding labels.
        Returns:
            - x_all: A list of SMILES strings that could be transformed into moleculer.
            - y_all: A list of the corresponding labels except NAN.
            - voc : A list of feature dictionary.
        """

        x_all, y_all = [], []
        for smiles, label in zip(x_smiles, labels):
            try:
                mol = MolFromSmiles(smiles)
                x_all.append(smiles)
                y_all.append(label)
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a RDKit Mol .\nREASON: {}'.format(smiles, e))

        self.voc = self.construct_vocabulary(x_all)

        return x_all, np.array(y_all), self.voc

    def construct_vocabulary(self, x_smiles):
        voc = set()
        for i, smiles in enumerate(x_smiles):
            smiles = smiles.split(" ")[0]
            regex = '(\[[^\[\]]{1,10}\])'
            smiles = self.replace_halogen(smiles)
            char_list = re.split(regex, smiles)
            for char in char_list:
                if char.startswith('['):
                    voc.add(char)
                else:
                    chars = [unit for unit in char]
                    [voc.add(unit) for unit in chars]

        return list(voc)

    def replace_halogen(self, string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        string = br.sub('R', string)
        string = cl.sub('L', string)
        return string

    def make_variables(self, lines, properties, all_letters):
        sequence_and_length = [self.line2voc_arr(line, all_letters) for line in lines]
        vectorized_seqs = [sl[0] for sl in sequence_and_length]
        seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
        return self.pad_sequences(vectorized_seqs, seq_lengths, properties)

    def letterToIndex(self, letter, all_letters):
        return all_letters.index(letter)

    def line2voc_arr(self, line, all_letters):
        arr = []
        regex = '(\[[^\[\]]{1,10}\])'
        line = self.replace_halogen(line.strip(' '))
        char_list = re.split(regex, line)
        for li, char in enumerate(char_list):
            if char.startswith('['):
                arr.append(self.letterToIndex(char, all_letters))
            else:
                chars = [unit for unit in char]

                for i, unit in enumerate(chars):
                    arr.append(self.letterToIndex(unit, all_letters))
        return arr, len(arr)

    # pad sequences and sort the tensor
    def pad_sequences(self, vectorized_seqs, seq_lengths, properties):
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        mask_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        # padding
        for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
            mask_tensor[idx, :] = torch.LongTensor(([1] * seq_len) + ([0] * (seq_lengths.max() - seq_len)))
        # Sort tensors by their length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        mask_tensor = mask_tensor[perm_idx]
        # Also sort the target (countries) in the same order
        # property_list = list(sorted(set(properties)))
        # property_ids = [property_list.index(property) for property in properties]
        target = torch.LongTensor(properties)
        if len(properties):
            target = target[perm_idx]
        
        return seq_tensor, target, mask_tensor