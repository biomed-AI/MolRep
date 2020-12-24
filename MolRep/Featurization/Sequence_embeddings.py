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

from MolRep.Utils.utils import *

class SequenceEmbeddings:
    def __init__(self, dataset_path, features_dir=None, use_data_saving=True, 
                       configs=None, dataset_configs=None, logger=None):
        """
        Args:
            - dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                                the first one contains SMILES strings of the compounds,
                                the second one contains labels.
            - features_dir (str): A path to save processed features.
            - use_data_saving (bool): If True, saved features will be loaded from the features_dir; if no feature file
                                    is present, the features will be saved after calculations. Defaults to True.
        """

        self.model_name = configs.model_name
        self.dataset_name = configs.dataset_name
        self.dataset_path = Path(dataset_path) / Path(dataset_configs["path"])
        self.features_dir = Path(dataset_path).parent if features_dir is None else Path(features_dir)

        self.use_data_saving = use_data_saving
        self.configs = configs
        self.logger = logger

        self.task_type = dataset_configs["task_type"]
        self.multi_class = self.task_type == 'Multiclass-Classification'
        self.multiclass_num_classes = dataset_configs["multiclass_num_classes"] if self.multi_class else None

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
            raise self.logger.error(f"Dataset filemat should be in CSV or SDF")

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
            self.logger.info(f"Processed features stored at '{output_path}'")

            _, _, _, voc = pickle.load(open(output_path, 'rb'))
            self.configs.dim_features = len(voc)

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            x_all, y_all, voc = self.load_data_from_smiles(data_x, data_y)

            seq_data, label, mask = self.make_variables(x_all, y_all, voc)
            self.configs.dim_features = len(voc)

            pickle.dump((seq_data, label, mask, voc), open(output_path, "wb"))
            self.logger.info(f"Saving features at '{output_path}'")

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

        voc = self.construct_vocabulary(x_all)

        return x_all, np.array(y_all), voc

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

