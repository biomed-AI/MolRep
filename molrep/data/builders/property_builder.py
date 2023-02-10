#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy, inc.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""

import os
import json
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from molrep.evaluations.data_split import scaffold_split, defined_split

from molrep.common.registry import registry
from molrep.data.builders.base_builder import BaseDatasetBuilder
from molrep.common.utils import filter_invalid_smiles, NumpyEncoder

from ogb.graphproppred import PygGraphPropPredDataset

@registry.register_builder("property_prediction")
class PropertyPredictionBuilder(BaseDatasetBuilder):

    model_processer_mapping = {
        "mpnn": "mpnn", "dmpnn": "mpnn", "cmpnn": "mpnn",
        "graphsage": "graph", "graphnet": "graph", "gin": "graph",
        "bilstm": "sequence", "salstm": "sequence", "transformer": "sequence",
    }

    def _download_and_load_data(self):
        dataset_path = Path(os.path.join(self.cache_root, self.dataset_config.storage[0]))

        if self.dataset_name.startswith('ogb'):
            if not os.path.exists(dataset_path):
                dataset = PygGraphPropPredDataset(self.dataset_name.replace('_', '-'), root=self.cache_root)
            self.whole_data_df = pd.read_csv(dataset_path, compression='gzip', header=0)

        elif dataset_path.suffix == '.csv':
            self.whole_data_df = pd.read_csv(dataset_path)
        
        elif dataset_path.suffix == '.sdf':
            self.whole_data_df = self._load_sdf_files(dataset_path)
        
        else:
            raise print(f"File Format must be in ['CSV', 'SDF'] or in OGB-Benchmark")

        self.smiles_col = self.config.datasets_cfg.smiles_column
        self.num_tasks = self.config.datasets_cfg.num_tasks
        self.multiclass_num_classes = self.config.datasets_cfg.multiclass_num_classes
        self.multi_class = self.multiclass_num_classes > 1

        valid_smiles = filter_invalid_smiles(list(self.whole_data_df.loc[:,self.smiles_col]))
        self.whole_data_df = self.whole_data_df[self.whole_data_df[self.smiles_col].isin(valid_smiles)].reset_index(drop=True)

        self._dim_targets = self.num_tasks * self.multiclass_num_classes if self.multi_class else self.num_tasks
        self.train_data_size = len(self.whole_data_df)

    def build(self):
        self.model_name = self.config.model_cfg.arch

        processor_cls = registry.get_processor_class(self.model_processer_mapping[self.model_name])(self.config, self.whole_data_df)
        processor_cls.process()

        self.config.datasets_cfg.dim_features = processor_cls.dim_features
        self.config.datasets_cfg.max_num_nodes = processor_cls.max_num_nodes
        self.config.datasets_cfg.dim_edge_features = processor_cls.dim_edge_features

        self.splits = self.setup_splits()
        datasets = self.construct_datasets()
        return datasets

    def construct_datasets(self):
        run_type = self.config.run_cfg.get("type", None)

        self.features_dir = Path(registry.get_path("features_root"))
        self.features_path = self.features_dir / f"{self.dataset_name}" / f"{self.model_name}.pt"

        if run_type == "train_test":
            datasets = {
                "train": None, "val": None, "test": None
            }

            indices = self.splits[0]["model_selection"][0]
            if 'validation' in indices.keys():
                trainset_indices = indices['train'] + indices['validation']
            else:
                trainset_indices = indices['train']
            testset_indices = self.splits[0]["test"]

            dataset_cls = registry.get_dataset_class(self.model_processer_mapping[self.model_name])
            train_dataset_cls = dataset_cls.construct_dataset(trainset_indices, self.features_path)
            datasets["train"] = train_dataset_cls.bulid_dataloader(self.config, is_train=True)
            test_dataset_cls = dataset_cls.construct_dataset(testset_indices, self.features_path)
            datasets["test"] = test_dataset_cls.bulid_dataloader(self.config, is_train=False)

        elif run_type == "train_val_test":
            datasets = {
                "train": None, "val": None, "test": None
            }

            indices = self.splits[0]["model_selection"][0]
            if 'validation' in indices.keys():
                trainset_indices = indices['train']
                validset_indices = indices['validation']
            else:
                trainset_indices = indices['train']
                validset_indices = []
            testset_indices = self.splits[0]['test']

            dataset_cls = registry.get_dataset_class(self.model_processer_mapping[self.model_name])

            train_dataset_cls = dataset_cls.construct_dataset(trainset_indices, self.features_path)
            datasets["train"] = train_dataset_cls.bulid_dataloader(self.config, is_train=True)

            valid_dataset_cls = dataset_cls.construct_dataset(validset_indices, self.features_path)
            datasets["val"] = valid_dataset_cls.bulid_dataloader(self.config, is_train=False)

            test_dataset_cls = dataset_cls.construct_dataset(testset_indices, self.features_path)
            datasets["test"] = test_dataset_cls.bulid_dataloader(self.config, is_train=False)

        elif run_type == "kfold":
            outer_folds = len(self.splits)
            datasets = {
                str(idx) : {"train": None, "val": None, "test": None}
                for idx in range(outer_folds)
            }

            for outer_k in range(outer_folds):
                indices = self.splits[outer_k]["model_selection"][0]
                trainset_indices = indices['train']
                testset_indices = self.splits[outer_k]["test"]

                dataset_cls = registry.get_dataset_class(self.model_processer_mapping[self.model_name])
                train_dataset_cls = dataset_cls.construct_dataset(trainset_indices, self.features_path)
                datasets[str(outer_k)]["train"] = train_dataset_cls.bulid_dataloader(self.config, is_train=True)
                test_dataset_cls = dataset_cls.construct_dataset(testset_indices, self.features_path)
                datasets[str(outer_k)]["test"] = test_dataset_cls.bulid_dataloader(self.config, is_train=False)

        return datasets

    def setup_splits(self):
        split_dir = Path(registry.get_path("split_root"))
        split_dir.mkdir(parents=True, exist_ok=True)
        splits_filename = split_dir / f"{self.dataset_name}_{self.dataset_config.split_type}_splits_seed{self.config.run_cfg.seed}.json"

        if splits_filename.exists():
            splits = json.load(open(splits_filename, "r"))

        elif self.dataset_name.startswith('ogb'):
            dataset = PygGraphPropPredDataset(self.dataset_name, root=self.cache_root)
            split_idx = dataset.get_idx_split()
            splits = [{"test": list(split_idx['test'].data.numpy()),
                            'model_selection': [{"train": list(split_idx['train'].data.numpy()) + list(split_idx['valid'].data.numpy()),
                                                 "validation": list(split_idx['test'].data.numpy())}]}]

            with open(splits_filename, "w") as f:
                json.dump(splits, f, cls=NumpyEncoder)

        else:
            splits = self._make_splits(splits_filename)

        return splits

    def _make_splits(self, splits_filename=None):
        """
        DISCLAIMER: train_test_split returns a SUBSET of the input indexes,
            whereas StratifiedKFold.split returns the indexes of the k subsets, starting from 0 to ...!
        """
        self.outer_k = self.config.run_cfg.outer_k
        self.inner_k = self.config.run_cfg.inner_k
        self.test_size = self.config.run_cfg.test_size
        self.validation_size = self.config.run_cfg.validation_size
        self.split_type = self.dataset_config.split_type

        assert (self.outer_k is not None and self.outer_k > 0) or self.outer_k is None
        assert (self.inner_k is not None and self.inner_k > 0) or self.inner_k is None

        self.seed = self.config.run_cfg.seed
        self.kfold_class = KFold if self.split_type == 'random' or self.split_type == 'defined' else StratifiedKFold

        splits = []

        all_idxs = np.arange(self.train_data_size)
        smiles = self.whole_data_df.loc[:,self.smiles_col].values
        targets = self.whole_data_df.loc[:,self.target_cols].values

        if 'additional_info' in self.dataset_config:
            self.defined_splits = self.whole_data_df[self.dataset_config["additional_info"]["splits"]].values

        if self.outer_k is None:  # holdout assessment strategy
            assert self.test_size is not None or 'additional_info' in self.dataset_config

            # Test-set splits
            if self.split_type == 'random':
                train_o_split, test_split = train_test_split(all_idxs,
                                                             test_size=self.test_size,
                                                             random_state=self.seed)

            elif self.split_type == 'stratified':
                train_o_split, test_split = train_test_split(all_idxs,
                                                             stratify=targets,
                                                             test_size=self.test_size,
                                                             random_state=self.seed)

            elif self.split_type == 'scaffold':
                train_o_split, test_split = scaffold_split(smiles,
                                                           test_size=self.test_size,
                                                           random_state=self.seed)

            elif self.split_type == 'defined':
                train_o_split, test_split = defined_split(self.defined_splits)

            else:
                assert f"{self.split_type} must be in [random, stratified, scaffold] or defined by yourself through dataset_config['additional_info']['splits']]."

            # Validation-set splits
            split = {"test": all_idxs[test_split], 'model_selection': []}
            train_o_smiles = smiles[train_o_split]
            train_o_targets = targets[train_o_split]

            if self.inner_k is None:  # holdout model selection strategy
                # if self.validation_size == 0:
                #     train_i_split, val_i_split = train_o_split, []
                # elif self.validation_size == -1:
                #     train_i_split, val_i_split = [], []
                # else:
                if self.split_type == 'random' or self.split_type == 'defined':
                    train_i_split, val_i_split = train_test_split(train_o_split,
                                                                  test_size=self.validation_size,
                                                                  random_state=self.seed)
                elif self.split_type == 'scaffold':
                    train_i_split, val_i_split = scaffold_split(train_o_smiles,
                                                                test_size=self.validation_size,
                                                                random_state=self.seed)
                elif self.split_type == 'stratified':
                    train_i_split, val_i_split = train_test_split(train_o_split,
                                                                  stratify=train_o_targets,
                                                                  test_size=self.validation_size,
                                                                  random_state=self.seed)
                elif self.split_type == 'defined':
                    train_i_split, val_i_split = defined_split(self.defined_splits, 'valid')
                else:
                    assert f"{self.split_type} must be in [random, stratified, scaffold] or defined by yourself."

                split['model_selection'].append(
                    {"train": train_i_split, "validation": val_i_split})

            else:  # cross validation model selection strategy
                if self.split_type == 'scaffold':
                    scaffold_test_size = (len(train_o_smiles) / self.inner_k) / len(train_o_smiles)
                    for _ in range(self.inner_k):
                        train_ik_split, val_ik_split = scaffold_split(train_o_smiles, test_size=scaffold_test_size,
                                                                        balanced=True, random_state=self.seed)
                        split['model_selection'].append(
                            {"train": train_ik_split, "validation": val_ik_split})

                elif self.split_type == 'stratified':
                    inner_kfold = self.kfold_class(n_splits=self.inner_k, shuffle=True, random_state=self.seed)
                    
                    for train_ik_split, val_ik_split in inner_kfold.split(train_o_split, train_o_targets):
                        split['model_selection'].append(
                            {"train": train_o_split[train_ik_split], "validation": train_o_split[val_ik_split]})

                elif self.split_type == 'random' or self.split_type == 'defined':
                    inner_kfold = self.kfold_class(n_splits=self.inner_k, shuffle=True, random_state=self.seed)
                
                    for train_ik_split, val_ik_split in inner_kfold.split(train_o_split):
                        split['model_selection'].append(
                            {"train": train_o_split[train_ik_split], "validation": train_o_split[val_ik_split]})

            splits.append(split)

        else:  # cross validation assessment strategy

            if self.split_type == 'scaffold':
                scaffold_test_size = (len(smiles) / self.outer_k) / len(smiles)
                for _ in range(self.outer_k):
                    train_ok_split, test_ok_split = scaffold_split(smiles, test_size=scaffold_test_size,
                                                                   balanced=True, random_state=self.seed)
                    split = {"test": all_idxs[test_ok_split], 'model_selection': []}

                    train_ok_smiles = smiles[train_ok_split]
                    train_ok_targets = targets[train_ok_split]
                    if self.inner_k is None:  # holdout model selection strategy
                        assert self.validation_size is not None
                        train_i_split, val_i_split = scaffold_split(train_ok_smiles,
                                                                    test_size=self.validation_size,
                                                                    random_state=self.seed)
                        split['model_selection'].append(
                            {"train": train_i_split, "validation": val_i_split})

                    else:
                        scaffold_test_size = (len(train_ok_smiles) / self.inner_k) / len(train_ok_smiles)
                        for inner_i in range(self.inner_k):
                            train_ik_split, val_ik_split = scaffold_split(train_ok_smiles, test_size=scaffold_test_size,
                                                                            balanced=True, random_state=self.seed)
                            split['model_selection'].append(
                                {"train": train_ik_split, "validation": val_ik_split})

                    splits.append(split)

            elif self.split_type == 'stratified':
                outer_kfold = self.kfold_class(n_splits=self.outer_k, shuffle=True, random_state=self.seed)
                
                for train_ok_split, test_ok_split in outer_kfold.split(all_idxs, targets):
                    split = {"test": all_idxs[test_ok_split], 'model_selection': []}

                    train_ok_smiles = smiles[train_ok_split]
                    train_ok_targets = targets[train_ok_split]
                    if self.inner_k is None:  # holdout model selection strategy
                        assert self.validation_size is not None
                        train_i_split, val_i_split = train_test_split(train_ok_split,
                                                                      stratify=train_ok_targets,
                                                                      test_size=self.validation_size,
                                                                      random_state=self.seed)
                        split['model_selection'].append(
                            {"train": train_i_split, "validation": val_i_split})

                    else:  # cross validation model selection strategy
                        inner_kfold = self.kfold_class(n_splits=self.inner_k, shuffle=True, random_state=self.seed)
                        for train_ik_split, val_ik_split in inner_kfold.split(train_ok_split, train_ok_targets):
                            split['model_selection'].append(
                                {"train": train_ok_split[train_ik_split], "validation": train_ok_split[val_ik_split]})

                    splits.append(split)

            elif self.split_type == 'random':
                outer_kfold = self.kfold_class(n_splits=self.outer_k, shuffle=True, random_state=self.seed)
                
                for train_ok_split, test_ok_split in outer_kfold.split(all_idxs):
                    split = {"test": all_idxs[test_ok_split], 'model_selection': []}

                    train_ok_smiles = smiles[train_ok_split]
                    train_ok_targets = targets[train_ok_split]
                    if self.inner_k is None:  # holdout model selection strategy
                        assert self.validation_size is not None
                        train_i_split, val_i_split = train_test_split(train_ok_split,
                                                                        test_size=self.validation_size,
                                                                        random_state=self.seed)
                        
                        split['model_selection'].append(
                            {"train": train_i_split, "validation": val_i_split})

                    else:  # cross validation model selection strategy
                        inner_kfold = self.kfold_class(n_splits=self.inner_k, shuffle=True, random_state=self.seed)
                        for train_ik_split, val_ik_split in inner_kfold.split(train_ok_split):
                            split['model_selection'].append(
                                {"train": train_ok_split[train_ik_split], "validation": train_ok_split[val_ik_split]})

                    splits.append(split)

            else:
                assert f"outer_k should be 'None' when split-type is 'defined'."

        with open(splits_filename, "w") as f:
            json.dump(splits, f, cls=NumpyEncoder)

        return splits

