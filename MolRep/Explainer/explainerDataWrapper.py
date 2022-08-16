# -*- coding: utf-8 -*-
"""
Created on 2021.08.19

@author: Jiahua Rao

"""


import os
import json
import numpy as np
import pandas as pd

from rdkit import Chem

from pathlib import Path
from sklearn.model_selection import train_test_split

from MolRep.Featurization.Graph_embeddings import GraphEmbeddings
from MolRep.Featurization.MPNN_embeddings import MPNNEmbeddings
from MolRep.Experiments.Graph_Data import Graph_data, MPNN_data
from MolRep.Evaluations.data_split import defined_split, scaffold_split

from MolRep.Utils.utils import filter_invalid_smiles, NumpyEncoder


class ExplainerDatasetWrapper:

    def __init__(self, dataset_config, model_name, seed=42, holdout_test_size=0.1,
                    split_dir='Splits', features_dir='Data'):


        self.dataset_config = dataset_config
        self.dataset_path = self.dataset_config["path"]
        self.dataset_name = self.dataset_config["name"]
        self.split_type = self.dataset_config["split_type"]
        self.attribution_path = self.dataset_config["attribution_path"] if 'attribution_path' in self.dataset_config else None
        self.model_name = model_name
        self.seed = seed
        self.holdout_test_size = holdout_test_size

        self._load_raw_data()
        self.features_dir = Path(features_dir)
        self.features_path = self.features_dir / f"{self.dataset_name}" / f"{self.model_name}.pt"

        self._max_num_nodes = None
        if not self.features_path.parent.exists():
            os.makedirs(self.features_path.parent)

        self._process()

        self.split_dir = Path(split_dir)
        self.splits_filename = self.split_dir / f"{self.dataset_name}_{self.split_type}_splits_seed{self.seed}.json"
        if not self.splits_filename.parent.exists():
            os.makedirs(self.splits_filename.parent)

        if not self.splits_filename.exists():
            self.splits = []
            self._make_splits()
        else:
            self.splits = json.load(open(self.splits_filename, "r"))


    @property
    def num_samples(self):
        return self.train_data_size

    @property
    def dim_features(self):
        return self._dim_features

    # @property
    # def dim_edge_features(self):
    #     return self._dim_edge_features

    @property
    def task_type(self):
        return self._task_type

    @property
    def dim_target(self):
        return self._dim_targets

    @property
    def max_num_nodes(self):
        return self._max_num_nodes

    def get_smiles_list(self, testing=True, training=False):
        whole_smiles = self.whole_data_df.loc[:,self.smiles_col].values
        if testing:
            testset_indices = self.splits[0]["test"]
        elif training:
            testset_indices = self.splits[0]["model_selection"][0]["train"]
        else:
            testset_indices = np.arange(len(whole_smiles))
        return whole_smiles[testset_indices]

    def get_smiles_idxs(self, testing=True, training=False):
        whole_smiles = self.whole_data_df.loc[:,self.smiles_col].values
        if testing:
            return self.splits[0]["test"]
        elif training:
            return self.splits[0]["model_selection"][0]["train"]
        else:
            return np.arange(len(whole_smiles))

    def get_attribution_truth(self):
        testset_indices = self.splits[0]["test"]
        attribution = np.load(self.attribution_path, allow_pickle=True)['attributions']

        if self.dataset_name in ['hERG', 'CYP3A4', 'CYP']:
            return attribution
        else:
            return [attribution[idx]['node_atts'] for idx in testset_indices]

    def _load_sdf_files(self, input_file, clean_mols=True):
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

    def _load_raw_data(self):

        self._task_type = self.dataset_config["task_type"]
        self.multi_class = self._task_type == 'Multi-Classification'
        self.multiclass_num_classes = self.dataset_config["multiclass_num_classes"] if self.multi_class else None

        self.smiles_col = self.dataset_config["smiles_column"]
        self.target_cols = self.dataset_config["target_columns"]
        self.num_tasks = len(self.target_cols)

        dataset_path = Path(self.dataset_path)
        if dataset_path.suffix == '.csv':
            self.whole_data_df = pd.read_csv(dataset_path)
        elif dataset_path.suffix == '.sdf':
            self.whole_data_df = self._load_sdf_files(dataset_path)
        else:
            raise print(f"File Format must be in ['CSV', 'SDF']")

        valid_smiles = filter_invalid_smiles(list(self.whole_data_df.loc[:,self.smiles_col]))
        self.whole_data_df = self.whole_data_df[self.whole_data_df[self.smiles_col].isin(valid_smiles)].reset_index(drop=True)

        self._dim_targets = self.num_tasks * self.multiclass_num_classes if self.multi_class else self.num_tasks
        self.train_data_size = len(self.whole_data_df)


    def _process(self):
        """
        """
        if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'GAT', 'PyGCMPNN', 'MorganFP', 'MACCSFP', 'XGraphSAGE', 'XGAT', 'XGIN', 'XMorganFP', 'XMACCSFP']:
            preparer = GraphEmbeddings(data_df=self.whole_data_df,
                                       model_name=self.model_name,
                                       features_path=self.features_path,
                                       dataset_config=self.dataset_config)
            preparer.process()
            self._dim_features = preparer.dim_features
            self._max_num_nodes = preparer.max_num_nodes
            
            if self.model_name in ['GraphNet', 'PyGCMPNN']:
                self._dim_features = (preparer.dim_features, preparer.dim_edge_features)


        elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            preparer = MPNNEmbeddings(data_df=self.whole_data_df,
                                      model_name=self.model_name,
                                      features_path=self.features_path,
                                      dataset_config=self.dataset_config)
            preparer.process()
            self._dim_features = preparer.dim_features
            self._max_num_nodes = preparer.max_num_nodes
            # self._dim_edge_features = preparer._dim_edge_features

        else: 
            raise print("Explainer Model Must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'PyGCMPNN', 'MPNN', 'DMPNN', 'CMPNN']")


    def _make_splits(self):
        """
        DISCLAIMER: train_test_split returns a SUBSET of the input indexes,
            whereas StratifiedKFold.split returns the indexes of the k subsets, starting from 0 to ...!
        """

        all_idxs = np.arange(self.train_data_size)
        smiles = self.whole_data_df.loc[:,self.smiles_col].values

        if 'additional_info' in self.dataset_config:
            self.defined_splits = self.whole_data_df[self.dataset_config["additional_info"]["splits"]].values
        else:
            self.defined_splits = np.repeat('train', len(self.whole_data_df))


        if self.split_type == 'random':
            train_split, test_split = train_test_split(all_idxs,
                                                            test_size=self.holdout_test_size,
                                                            random_state=self.seed)

        elif self.split_type == 'scaffold':
            train_split, test_split = scaffold_split(smiles,
                                                        test_size=self.holdout_test_size,
                                                        random_state=self.seed)

        elif self.split_type == 'defined':
            _, test_split = defined_split(self.defined_splits)
            train_split, valid_split = defined_split(self.defined_splits, 'valid')

        else:
            assert f"{self.split_type} must be in [random, scaffold] or defined by yourself through dataset_config['additional_info']['splits']]."

        split = {"test": all_idxs[test_split], 'model_selection': []}
        split['model_selection'].append(
                                {"train": all_idxs[train_split], "valid":all_idxs[valid_split]})
        self.splits.append(split)

        with open(self.splits_filename, "w") as f:
            json.dump(self.splits, f, cls=NumpyEncoder)


    def get_train_loader(self, batch_size=1, shuffle=True, features_scaling=False):

        indices = self.splits[0]["model_selection"][0]
        trainset_indices = indices['train']
        validset_indices = indices['valid'] if len(indices['valid']) else None

        scaler = None
        if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'GAT', 'PyGCMPNN', 'MorganFP', 'MACCSFP', 'XGraphSAGE', 'XGAT', 'XGIN', 'XMorganFP', 'XMACCSFP']:
            train_dataset, valid_dataset, _ = Graph_data.Graph_construct_dataset(
                self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            train_loader, valid_loader, _, features_scaler, scaler = Graph_data.Graph_construct_dataloader(
                trainset=train_dataset, validset=valid_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            train_dataset, valid_dataset, _ = MPNN_data.MPNN_construct_dataset(
                self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            
            train_loader, valid_loader, _, features_scaler, scaler = MPNN_data.MPNN_construct_dataloader(
                trainset=train_dataset, validset=valid_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)
            
        else: 
            raise print(self.model_name, " Explainer Model Must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'PyGCMPNN', 'MPNN', 'DMPNN', 'CMPNN']")

        return train_loader, valid_loader, features_scaler, scaler


    def get_test_loader(self, batch_size=1, features_scaling=False):

        testset_indices = self.splits[0]["test"]

        if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'GAT', 'PyGCMPNN', 'MorganFP', 'MACCSFP', 'XGraphSAGE', 'XGAT', 'XGIN', 'XMorganFP', 'XMACCSFP']:
            _, _, test_dataset = Graph_data.Graph_construct_dataset(
                                                    self.features_path, test_idxs=testset_indices)
            _, _, test_loader, _, _ = Graph_data.Graph_construct_dataloader(
                                                    testset=test_dataset, batch_size=batch_size, shuffle=False, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            _, _, test_dataset = MPNN_data.MPNN_construct_dataset(
                                                    self.features_path, test_idxs=testset_indices)
            _, _, test_loader, _, _ = MPNN_data.MPNN_construct_dataloader(
                                                        testset=test_dataset, batch_size=batch_size, shuffle=False, task_type=self._task_type, features_scaling=features_scaling)

        else: 
            raise print("Explainer Model Must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'PyGCMPNN', 'MPNN', 'DMPNN', 'CMPNN']")

        if len(testset_indices) == 0:
            test_loader = None
        return test_loader

    def get_all_dataloader(self, batch_size=1, shuffle=False, features_scaling=False):

        indices = self.splits[0]["model_selection"][0]
        testset_indices = indices['train']
        testset_indices += self.splits[0]["test"]

        if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'GAT', 'PyGCMPNN', 'MolecularFingerprint']:
            _, _, test_dataset = Graph_data.Graph_construct_dataset(
                                                    self.features_path, test_idxs=testset_indices)
            _, _, test_loader, _, _ = Graph_data.Graph_construct_dataloader(
                                                    testset=test_dataset, batch_size=batch_size, shuffle=False, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            _, _, test_dataset = MPNN_data.MPNN_construct_dataset(
                                                    self.features_path, test_idxs=testset_indices)
            _, _, test_loader, _, _ = MPNN_data.MPNN_construct_dataloader(
                                                        testset=test_dataset, batch_size=batch_size, shuffle=False, task_type=self._task_type, features_scaling=features_scaling)

        else: 
            raise print("Explainer Model Must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'PyGCMPNN', 'MPNN', 'DMPNN', 'CMPNN']")

        if len(testset_indices) == 0:
            test_loader = None
        return test_loader