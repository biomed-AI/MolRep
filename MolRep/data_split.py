# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie
'''

import os
import math
import json
import random
import argparse

import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, MolFromSmiles, MolFromMolBlock, MolToSmarts

from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from MolRep.Utils.utils import filter_invalid_smiles

class DataSplit():
    def __init__(self, dataset_path, dataset_name, split_dir=None, n_splits=5,
                 shuffle=True, seed=0, split_size=[0.8, 0.1, 0.1],
                 dataset_configs=None, logger=None):
        """
        Args:
            - dataset_path (str): A path to save datasets. (.csv or .sdf)
            - dataset_name (str): A name of the dataset.
            - split_dir (str): A path to save data splits.
            - n_splits (int): Number of folds. Must be at least 2.
            - shuffle (bool): If True the data will be loaded in a random order. Defaults to True.
            - seed (int): Seed for shuffling when doing splitting.
            - split_size (list[float]): Split proportions for train/validation/test sets.
            - dataset_configs (dict): Namespace of dataset configuration.
            - logger (logging): logging.
        """

        self.dataset_configs = dataset_configs

        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path) / self.dataset_configs["path"]
        self.split_dir = Path(dataset_path).parent if split_dir is None else Path(split_dir)

        self.n_splits = n_splits
        self.split_size = split_size
        self.seed = seed
        self.shuffle = shuffle
        self.logger = logger

        self.smiles_col = self.dataset_configs["smiles_column"]
        self.target_cols = self.dataset_configs["target_columns"]

        if self.dataset_path.suffix == '.csv':
            self.whole_data_df = pd.read_csv(self.dataset_path)
        elif self.dataset_path.suffix == '.sdf':
            self.whole_data_df = self.load_sdf_files(self.dataset_path)

        valid_smiles = filter_invalid_smiles(list(self.whole_data_df.loc[:,self.smiles_col]))
        self.whole_data_df = self.whole_data_df[self.whole_data_df[self.smiles_col].isin(valid_smiles)].reset_index(drop=True)

        self.output_dir = self.split_dir #/ self.dataset_name
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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

    def scaffold_split(self, balanced=True, use_indices=True, include_chirality=False):
        """
        Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.
        Args:
            - balanced (bool): Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
            - use_indices (bool): Whether to map to the smiles' index in all_smiles rather than mapping
                                  to the smiles string itself. This is necessary if there are duplicate smiles.
            - include_chirality (bool): Whether to include chirality.

        return: 
            A dictionary containing N-fold splits of the data.
        """
        n_splits, seed = self.n_splits, self.seed

        output_path = self.output_dir / f"{self.dataset_name}_scaffold_splits_seed{self.seed}.json"

        if os.path.exists(output_path):
            self.logger.info(f"Processed data split existed.")
            self.logger.info(f"Loading data split stored at '{output_path}'")
        else:
            data = self.whole_data_df
            fold_size = math.ceil(len(data) / n_splits)
            scaffold_to_indices = self.scaffold_to_smiles(list(data.loc[:,self.smiles_col]), use_indices=use_indices, include_chirality=include_chirality)
            if balanced:
            # Put stuff that's bigger than half the val/test size into train, rest just order randomly
                index_sets = list(scaffold_to_indices.values())
                big_index_sets = []
                small_index_sets = []
                for index_set in index_sets:
                    if len(index_set) > fold_size / 2:
                        big_index_sets.append(index_set)
                    else:
                        small_index_sets.append(index_set)
                random.seed(seed)
                random.shuffle(big_index_sets)
                random.shuffle(small_index_sets)
                index_sets = big_index_sets + small_index_sets
            else:
            # Sort from largest to smallest scaffold sets
                index_sets = sorted(list(scaffold_to_indices.values()),
                                    key=lambda index_set: len(index_set),
                                    reverse=True)

            n_splits_data, fold_index = [], {}
            for k in range(n_splits):
                fold_index['fold_%d'%(k+1)] = []

            for index_set in index_sets:
                for k in range(n_splits):
                    if len(fold_index['fold_%d'%(k+1)]) + len(index_set) <= fold_size:
                        fold_index['fold_%d'%(k+1)] += index_set
                        break
                    if k == (n_splits - 1):
                        fold_index['fold_%d'%(k+1)] += index_set

            for k in range(n_splits):
                test_index = fold_index['fold_%d'%(k+1)]
                train_index = []
                for i in range(n_splits):
                    if i != k: train_index = train_index + fold_index['fold_%d'%(i+1)]
                n_splits_data.append({'train':train_index, 'test':test_index})

            self.logger.info(f"Saving data split at '{output_path}' by scaffold.")
            self.write_to_json(n_splits_data, output_path)

    def scaffold_test(self, balanced=True, use_indices=True, include_chirality=False):
        """
        Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.
        Args:
            - balanced (bool): Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
            - use_indices (bool): Whether to map to the smiles' index in all_smiles rather than mapping
                                  to the smiles string itself. This is necessary if there are duplicate smiles.
            - include_chirality (bool): Whether to include chirality.

        return: 
            A dictionary containing train/valid/test splits of the data.
        """
        assert sum(self.split_size) == 1

        output_path = self.output_dir / f"{self.dataset_name}_scaffold_test_splits_seed{self.seed}.json"

        if os.path.exists(output_path):
            self.logger.info(f"Processed data split existed.")
            self.logger.info(f"Loading data split stored at '{output_path}'")
        else:
            data = self.whole_data_df
            size = self.split_size
            # Split
            train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
            train, val, test = [], [], []
            train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

            scaffold_to_indices = self.scaffold_to_smiles(list(data.loc[:,self.smiles_col]), use_indices=use_indices, include_chirality=include_chirality)
            if balanced:
            # Put stuff that's bigger than half the val/test size into train, rest just order randomly
                index_sets = list(scaffold_to_indices.values())
                big_index_sets = []
                small_index_sets = []
                for index_set in index_sets:
                    if len(index_set) > fold_size / 2:
                        big_index_sets.append(index_set)
                    else:
                        small_index_sets.append(index_set)
                random.seed(seed)
                random.shuffle(big_index_sets)
                random.shuffle(small_index_sets)
                index_sets = big_index_sets + small_index_sets
            else:
            # Sort from largest to smallest scaffold sets
                index_sets = sorted(list(scaffold_to_indices.values()),
                                    key=lambda index_set: len(index_set),
                                    reverse=True)

            n_splits_data  = []
            for index_set in index_sets:
                if len(train) + len(index_set) <= train_size:
                    train += index_set
                    train_scaffold_count += 1
                elif len(val) + len(index_set) <= val_size:
                    val += index_set
                    val_scaffold_count += 1
                else:
                    test += index_set
                    test_scaffold_count += 1
            val = None if len(val) == 0 else val
            n_splits_data.append({'train':train, 'valid':val, 'test':test})

            self.logger.info(f"Saving data split at '{output_path}' by scaffold.")
            self.write_to_json(n_splits_data, output_path)


    def generate_scaffold(self, smiles, include_chirality=False):
        """
        Compute the Bemis-Murcko scaffold for a SMILES string.
        Args:
            smiles (str): A smiles string.
        
        return:
            Bemis-Murcko scaffold
        """
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold

    def scaffold_to_smiles(self, mols, use_indices=True, include_chirality=False):
        """
        Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.
        Args:
            - mols (rdchem.Mol): A list of smiles strings or RDKit molecules.

        return: 
            A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
        """
        scaffolds = defaultdict(set)
        for i, mol in enumerate(mols):
            scaffold = self.generate_scaffold(mol, include_chirality=include_chirality)
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)
        return scaffolds


    def stratified_kfold(self):
        """
        Split a dataset by stratified_kfold.
        Args:
            - n_splits (int): Number of folds. Must be at least 2.

        return: 
            A dictionary containing N-fold splits of the data.
        """

        assert len(self.target_cols) == 1

        output_path = self.output_dir / f"{self.dataset_name}_stratified-kfold_splits_seed{self.seed}.json"

        if os.path.exists(output_path):
            self.logger.info(f"Processed data split existed.")
            self.logger.info(f"Loading data split stored at '{output_path}'")
        else:            
            data = self.whole_data_df
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
            n_splits_data, fold_index = [], {}
            for _, (train_index, test_index) in enumerate(skf.split(data.loc[:,self.smiles_col], data.loc[:,self.target_cols])):
                fold_index['train'] = [int(i) for i in list(train_index)]
                fold_index['test'] = [int(i) for i in list(test_index)]
                n_splits_data.append(fold_index)

            self.logger.info(f"Saving data split at '{output_path}' by stratified-kfold.")
            self.write_to_json(n_splits_data, output_path)


    def kfold(self):
        """
        Split a dataset by kfold (random).

        return: 
            A dictionary containing N-fold splits of the data.
        """
        output_path = self.output_dir / f"{self.dataset_name}_random_splits_seed{self.seed}.json"
        if os.path.exists(output_path):
            self.logger.info(f"Processed data split existed.")
            self.logger.info(f"Loading data split stored at '{output_path}'")
        else:
            data = self.whole_data_df
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
            n_splits_data, fold_index = [], {}
            for _, (train_index, test_index) in enumerate(kf.split(data.loc[:,self.smiles_col])):
                fold_index['train'] = [int(i) for i in list(train_index)]
                fold_index['test'] = [int(i) for i in list(test_index)]
                n_splits_data.append(fold_index)

            self.logger.info(f"Saving data split at '{output_path}' by kfold.")
            self.write_to_json(n_splits_data, output_path)

    def random_test(self):
        """
        Split a dataset by kfold (random).

        return: 
            A dictionary containing train/valid/test splits of the data.
        """
        assert sum(self.split_size) == 1

        output_path = self.output_dir / f"{self.dataset_name}_random_test_splits_seed{self.seed}.json"
        if os.path.exists(output_path):
            self.logger.info(f"Processed data split existed.")
            self.logger.info(f"Loading data split stored at '{output_path}'")
        else:
            data = self.whole_data_df
            all_indices = np.arange(len(data))

            train_val_indices, test_index = train_test_split(all_indices, test_size=self.split_size[-1], random_state=self.seed)
            if self.split_size[1] != 0.0:
                train_index, valid_index = train_test_split(train_val_indices, test_size=self.split_size[1], random_state=self.seed)
            else:
                train_index, valid_index = train_val_indices, None

            n_splits_data = []
            n_splits_data.append({'train':train_index, 'valid':valid_index, 'test':test_index})

            self.logger.info(f"Saving data split at '{output_path}' by random.")
            self.write_to_json(n_splits_data, output_path)

    def write_to_json(self, n_splits_data, output_path):
        with open(output_path, 'w') as fp:
            json.dump(n_splits_data, fp)


def k_fold_data_splits(seed, configs, dataset_configs=None, logger=None):
    """
    Data splits by different ways, including scaffold, random(k-fold) and stratified-kfold.
    Args:
        - seed (int): Seed for shuffling when doing splitting.
        - configs (Namespace): Namespace of basic configuration.
        - dataset_configs (dict): Namespace of dataset configuration.
        - logger (logging): logging.
    """

    Data_split = DataSplit(dataset_path=configs.dataset_path, dataset_name=configs.dataset_name,
                           split_dir=configs.split_dir, n_splits=configs.k_fold, shuffle=configs.shuffle,
                           seed=seed, split_size=configs.split_sizes, dataset_configs=dataset_configs, logger=logger)

    if dataset_configs["split_type"] == 'stratified-kfold' and configs.k_fold is not None:
        Data_split.stratified_kfold()

    elif dataset_configs["split_type"] == 'scaffold' and configs.k_fold is not None:
        Data_split.scaffold_split()

    elif dataset_configs["split_type"] == 'random' and configs.k_fold is not None:
        Data_split.kfold()

    elif dataset_configs["split_type"] == 'scaffold':
        Data_split.scaffold_test()

    elif dataset_configs["split_type"] == 'random':
        Data_split.random_test()

    else:
        raise self.logger.error(f"Data split Type must be in ['stratified-kfold', 'random', 'scaffold'].")