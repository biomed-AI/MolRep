"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on: 
Zheng, Shuangjia, et al. "Identifying structure–property relationships through SMILES syntax analysis with self-attention mechanism." Journal of chemical information and modeling 59.2 (2019): 914-923.
"""


import os, re
import pickle
import numpy as np

import torch

from pathlib import Path
from rdkit.Chem import MolFromSmiles

from MolRep.Featurization.utils.seq_utils import *

class SequenceEmbeddings:
    def __init__(self, data_df, model_name, features_path, dataset_config,
                 use_data_saving=True, max_length=0):

        self.model_name = model_name
        self.whole_data_df = data_df
        self.features_path = features_path
        self.dataset_config = dataset_config

        self.use_data_saving = use_data_saving
        self.smiles_col = self.dataset_config["smiles_column"]
        self.target_cols = self.dataset_config["target_columns"]

        vocab_filepath = Path(dataset_config['path']) / "vocab.txt"
        self.tokenizer = SmilesTokenizer(vocab_file=vocab_filepath)
        
        self.max_length = max_length if max_length else self.get_max_length()

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
            # seq_data, label, mask, voc = pickle.load(open(features_path, 'rb'))
            dataset = torch.load(features_path)
            # dataset = (seq_data, label, mask, voc)
            self._dim_features = len(dataset["voc"])

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            x_all, y_all, voc = self.load_data_from_smiles(data_x, data_y)

            seq_data, label, mask = self.make_variables(x_all, y_all)
            self._dim_features = len(voc)

            # pickle.dump((seq_data, label, mask, voc), open(features_path, "wb"))
            dataset = {
                "mask": mask,
                "voc": voc,
                "label": label,
                "seq_data": seq_data,
            }
            torch.save(dataset, features_path)

        return dataset

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

        voc = self.tokenizer.vocab

        return x_all, np.array(y_all), voc

    def make_variables(self, smiles, labels):
        input_ids, mask = [], []
        for i, smi in enumerate(smiles):
            smi_input_ids, smi_mask = self.tokenize_smiles(smi)
            input_ids.append(smi_input_ids)
            mask.append(smi_mask)
        input_ids = torch.stack(input_ids, dim=0)
        mask = torch.stack(mask, dim=0)
        labels = torch.LongTensor(labels)
        return input_ids, labels, mask

    def tokenize_smiles(self, smiles):
        encoded_inputs = self.tokenizer(smiles, max_length=self.max_length, padding='max_length', truncation=True)
        smiles_input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return smiles_input_ids, mask

    def get_max_length(self):
        smiles_list = self.whole_data_df.loc[:,self.smiles_col].tolist()
        max_length = 0
        for i, smi in enumerate(smiles_list):
            token_list = self.tokenizer._tokenize(smi.strip(" "))
            if len(token_list) >= max_length:
                max_length = len(token_list)
        return max_length