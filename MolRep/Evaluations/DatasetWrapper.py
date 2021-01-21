
import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from MolRep.Evaluations.data_split import specific_split

from MolRep.Featurization.Graph_embeddings import GraphEmbeddings
from MolRep.Featurization.MPNN_embeddings import MPNNEmbeddings

from MolRep.Featurization.MAT_embeddings import MATEmbeddings
from MolRep.Featurization.CoMPT_embeddings import CoMPTEmbeddings
from MolRep.Featurization.VAE_embeddings import VAEEmbeddings
from MolRep.Featurization.Sequence_embeddings import SequenceEmbeddings

from MolRep.Experiments.Graph_Data import Graph_data, MPNN_data
from MolRep.Experiments.Sequence_Data import Sequence_data, MAT_data, CoMPT_data
from MolRep.Experiments.Unsupervised_Data import VAE_data

from MolRep.Utils.utils import filter_invalid_smiles, NumpyEncoder


class DatasetWrapper:

    def __init__(self, dataset_config, model_name, kfold_class=KFold, outer_k=10, inner_k=None, seed=42, holdout_test_size=0.1, 
                    split_dir='MolRep/Splits', features_dir='MolRep/Data'):

        self.dataset_config = dataset_config
        self.dataset_path = self.dataset_config["path"]
        self.dataset_name = self.dataset_config["name"]
        self.split_type = self.dataset_config["split_type"]
        self.model_name = model_name

        self.kfold_class = kfold_class
        self.outer_k = outer_k
        self.inner_k = inner_k
        self.holdout_test_size = holdout_test_size

        self.outer_k = outer_k
        assert (outer_k is not None and outer_k > 0) or outer_k is None

        self.inner_k = inner_k
        assert (inner_k is not None and inner_k > 0) or inner_k is None

        self.seed = seed

        self._load_raw_data()

        self.features_dir = Path(features_dir)
        self.features_path = self.features_dir / f"{self.dataset_name}" / f"{self.model_name}.pt"

        self._max_num_nodes = None
        if not self.features_path.parent.exists():
            os.makedirs(self.features_path.parent)
        self._process()

        self.split_dir = Path(split_dir)
        self.splits_filename = self.split_dir / self.dataset_name / f"{self.model_name}_{self.split_type}_splits_seed{self.seed}.json"
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

    @property
    def task_type(self):
        return self._task_type

    @property
    def dim_target(self):
        return self._dim_targets

    @property
    def max_num_nodes(self):
        return self._max_num_nodes

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
        self.multi_class = self._task_type == 'Multiclass-Classification'
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
        if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MolecularFingerprint']:
            preparer = GraphEmbeddings(data_df=self.whole_data_df,
                                       model_name=self.model_name,
                                       features_path=self.features_path,
                                       dataset_config=self.dataset_config)
            preparer.process()
            self._dim_features = preparer.dim_features
            self._max_num_nodes = preparer.max_num_nodes

        elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            preparer = MPNNEmbeddings(data_df=self.whole_data_df,
                                      model_name=self.model_name,
                                      features_path=self.features_path,
                                      dataset_config=self.dataset_config)
            preparer.process()
            self._dim_features = preparer.dim_features
            self._max_num_nodes = preparer.max_num_nodes

        elif self.model_name == 'MAT':
            preparer = MATEmbeddings(data_df=self.whole_data_df,
                                     model_name=self.model_name,
                                     features_path=self.features_path,
                                     dataset_config=self.dataset_config)
            preparer.process()
            self._dim_features = preparer.dim_features
            self._max_num_nodes = preparer.max_num_nodes

        elif self.model_name == 'CoMPT':
            preparer = CoMPTEmbeddings(data_df=self.whole_data_df,
                                       model_name=self.model_name,
                                       features_path=self.features_path,
                                       dataset_config=self.dataset_config)
            preparer.process()
            self._dim_features = preparer.dim_features
            self._max_num_nodes = preparer.max_num_nodes

        elif self.model_name in ['BiLSTM', 'SALSTM', 'Transformer']:
            preparer = SequenceEmbeddings(data_df=self.whole_data_df,
                                          model_name=self.model_name,
                                          features_path=self.features_path,
                                          dataset_config=self.dataset_config)
            preparer.process()
            self._dim_features = preparer.dim_features
            self._max_num_nodes = preparer.max_num_nodes

        elif self.model_name == 'VAE':
            preparer = VAEEmbeddings(data_df=self.whole_data_df,
                                     model_name=self.model_name,
                                     features_path=self.features_path,
                                     dataset_config=self.dataset_config)
            preparer.process()
            self._dim_features = preparer.dim_features
            self._max_num_nodes = preparer.max_num_nodes


    def _make_splits(self):
        """
        DISCLAIMER: train_test_split returns a SUBSET of the input indexes,
            whereas StratifiedKFold.split returns the indexes of the k subsets, starting from 0 to ...!
        """

        all_idxs = np.arange(self.train_data_size)
        smiles = self.whole_data_df.loc[:,self.smiles_col].values
        targets = self.whole_data_df.loc[:,self.target_cols].values

        if 'additional_info' in self.dataset_config:
            self.cell_types = self.whole_data_df[self.dataset_config["additional_info"]["cell_type_column"]].values

        if self.outer_k is None:  # holdout assessment strategy
            assert self.holdout_test_size is not None

            if self.holdout_test_size == 0:
                train_o_split, test_split = all_idxs, []
            elif self.holdout_test_size == -1:
                train_o_split, test_split = [], all_idxs
            else:
                if self.split_type == 'random':
                    train_o_split, test_split = train_test_split(all_idxs,
                                                                 test_size=self.holdout_test_size,
                                                                 random_state=self.seed)
                elif self.split_type == 'stratified':
                    train_o_split, test_split = train_test_split(all_idxs,
                                                                 stratify=targets,
                                                                 test_size=self.holdout_test_size,
                                                                 random_state=self.seed)
                # elif self.split_type == 'scaffold':
                #     train_o_split, test_split = scaffold_split(smiles,
                #                                                test_size=self.holdout_test_size,
                #                                                random_state=self.seed)
                elif self.split_type == 'specific':
                    train_o_split, test_split = specific_split(all_idxs,
                                                               cell_type=self.cell_types,
                                                               test_size=self.holdout_test_size,
                                                               random_state=self.seed)


            split = {"test": all_idxs[test_split], 'model_selection': []}
            train_o_smiles = smiles[train_o_split]
            train_o_targets = targets[train_o_split]

            if self.inner_k is None:  # holdout model selection strategy
                if self.holdout_test_size == 0:
                    train_i_split, val_i_split = train_o_split, []
                elif self.holdout_test_size == -1:
                    train_i_split, val_i_split = [], []
                else:
                    if self.split_type == 'random':
                        train_i_split, val_i_split = train_test_split(train_o_split,
                                                                      test_size=self.holdout_test_size,
                                                                      random_state=self.seed)
                    # elif self.split_type == 'scaffold':
                    #     train_i_split, val_i_split = scaffold_split(train_o_split,
                    #                                                 test_size=self.holdout_test_size,
                    #                                                 random_state=self.seed)
                    elif self.split_type == 'specific':
                        train_i_split, val_i_split = specific_split(train_o_split,
                                                                    test_size=self.holdout_test_size,
                                                                    random_state=self.seed)

                split['model_selection'].append(
                    {"train": train_i_split, "validation": val_i_split})

            else:  # cross validation model selection strategy
                inner_kfold = self.kfold_class(n_splits=self.inner_k, shuffle=True, random_state=self.seed)
                for train_ik_split, val_ik_split in inner_kfold.split(train_o_split):
                    split['model_selection'].append(
                        {"train": train_o_split[train_ik_split], "validation": train_o_split[val_ik_split]})

            self.splits.append(split)

        else:  # cross validation assessment strategy
            outer_kfold = self.kfold_class(n_splits=self.outer_k, shuffle=True, random_state=self.seed)

            for train_ok_split, test_ok_split in outer_kfold.split(all_idxs):
                split = {"test": all_idxs[test_ok_split], 'model_selection': []}

                if self.inner_k is None:  # holdout model selection strategy
                    assert self.holdout_test_size is not None
                    if self.split_type == 'random':
                        train_i_split, val_i_split = train_test_split(train_ok_split,
                                                                      test_size=self.holdout_test_size,
                                                                      random_state=self.seed)
                    # else:
                    #     train_i_split, val_i_split = scaffold_split(train_ok_split,
                    #                                                 test_size=self.holdout_test_size,
                    #                                                 random_state=self.seed)

                    split['model_selection'].append(
                        {"train": train_i_split, "validation": val_i_split})

                else:  # cross validation model selection strategy
                    inner_kfold = self.kfold_class(n_splits=self.inner_k, shuffle=True, random_state=self.seed)
                    for train_ik_split, val_ik_split in inner_kfold.split(train_ok_split):
                        split['model_selection'].append(
                            {"train": train_ok_split[train_ik_split], "validation": train_ok_split[val_ik_split]})

                self.splits.append(split)

        with open(self.splits_filename, "w") as f:
            json.dump(self.splits, f, cls=NumpyEncoder)

    def get_test_fold(self, outer_idx, batch_size=1, shuffle=True, features_scaling=False):
        outer_idx = outer_idx or 0

        testset_indices = self.splits[outer_idx]["test"]

        if len(test_data) == 0:
            test_loader = None

        else:

            if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MolecularFingerprint']:
                _, _, test_dataset = Graph_data.Graph_construct_dataset(
                                                        self.features_path, test_idxs=testset_indices)
                _, _, test_loader, _, _ = Graph_data.Graph_construct_dataloader(
                                                        testset=test_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

            elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
                _, _, test_dataset = MPNN_data.MPNN_construct_dataset(
                                                        self.features_path, test_idxs=testset_indices)
                _, _, test_loader, _, _ = MPNN_data.MPNN_construct_dataloader(
                                                        testset=test_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

            elif self.model_name == 'MAT':
                _, _, test_dataset = MAT_data.MAT_construct_dataset(
                                                        self.features_path, test_idxs=testset_indices)
                _, _, test_loader, _, _ = MAT_data.MAT_construct_loader(
                                                        testset=test_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)
            
            elif self.model_name == 'CoMPT':
                _, _, test_dataset = CoMPT_data.CoMPT_construct_dataset(
                                                        self.features_path, test_idxs=testset_indices)
                _, _, test_loader, _, _ = CoMPT_data.CoMPT_construct_loader(
                                                        testset=test_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

            elif self.model_name in ['BiLSTM', 'SALSTM', 'Transformer']:
                _, _, test_dataset = Sequence_data.Sequence_construct_dataset(
                                                        self.features_path, test_idxs=testset_indices)
                _, _, test_loader, _, _ = Sequence_data.Sequence_construct_loader(
                                                        testset=test_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

            elif self.model_name == 'VAE':
                _, _, test_dataset = VAE_data.VAE_construct_dataset(
                                                        self.features_path, test_idxs=testset_indices)
                _, _, test_loader, _, _ = VAE_data.VAE_construct_loader(
                                                        testset=test_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

            elif self.model_name in ['N_Gram_Graph', 'Mol2Vec']:
                if self.model_name == 'Mol2Vec':
                    _, _, test_loader, _, _ = Mol2Vec_data.Mol2Vec_construct_loader(
                                                        self.features_path, test_idxs=testset_indices)
                elif self.model_name == 'N_Gram_Graph':
                    _, _, test_loader, _, _ = NGramGraph_data.N_Gram_Graph_construct_loader(
                                                        self.features_path, test_idxs=testset_indices)
            else:
                raise self.logger.error(f"Model Name must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MolecularFingerprint', \
                                                'MPNN', 'DMPNN', 'CMPNN', 'MAT', 'BiLSTM', 'BiLSTM-Attention']")

        return test_loader


    def get_model_selection_fold(self, outer_idx, inner_idx=None, batch_size=1, shuffle=True, features_scaling=False):
        outer_idx = outer_idx or 0

        if inner_idx is not None:
            indices = self.splits[outer_idx]["model_selection"][inner_idx]
            trainset_indices = indices['train']
            validset_indices = indices['validation']
        else:
            indices = self.splits[outer_idx]["model_selection"][inner_idx]
            trainset_indices = indices['train'] + indices['validation']
            validset_indices = []

        scaler = None

        if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MolecularFingerprint']:
            train_dataset, valid_dataset, _ = Graph_data.Graph_construct_dataset(
                self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            train_loader, valid_loader, _, features_scaler, scaler = Graph_data.Graph_construct_dataloader(
                trainset=train_dataset, validset=valid_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            train_dataset, valid_dataset, _ = MPNN_data.MPNN_construct_dataset(
                self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            
            train_loader, valid_loader, _, features_scaler, scaler = MPNN_data.MPNN_construct_dataloader(
                trainset=train_dataset, validset=valid_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name == 'MAT':
            train_dataset, valid_dataset, _ = MAT_data.MAT_construct_dataset(
                    self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            train_loader, valid_loader, _, features_scaler, scaler = MAT_data.MAT_construct_loader(
                    trainset=train_dataset, validset=valid_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name == 'CoMPT':
            train_dataset, valid_dataset, _ = CoMPT_data.CoMPT_construct_dataset(
                    self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            train_loader, valid_loader, _, features_scaler, scaler = CoMPT_data.CoMPT_construct_loader(
                    trainset=train_dataset, validset=valid_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name in ['BiLSTM', 'SALSTM', 'Transformer']:
            train_dataset, valid_dataset, _ = Sequence_data.Sequence_construct_dataset(
                self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            train_loader, valid_loader, _, features_scaler, scaler = Sequence_data.Sequence_construct_loader(
                trainset=train_dataset, validset=valid_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name == 'VAE':
            train_dataset, valid_dataset, _ = VAE_data.VAE_construct_dataset(
                self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            train_loader, valid_loader, _, features_scaler, scaler = VAE_data.VAE_construct_loader(
                trainset=train_dataset, validset=valid_dataset, batch_size=batch_size, shuffle=shuffle, task_type=self._task_type, features_scaling=features_scaling)

        elif self.model_name in ['N_Gram_Graph', 'Mol2Vec']:
            if self.model_name == 'Mol2Vec':
                train_loader, valid_loader, _, features_scaler, scaler = Mol2Vec_data.Mol2Vec_construct_loader(
                        self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
            elif self.model_name == 'N_Gram_Graph':
                train_loader, valid_loader, _, features_scaler, scaler = NGramGraph_data.N_Gram_Graph_construct_loader(
                        self.features_path, train_idxs=trainset_indices, valid_idxs=validset_indices)
        else:
            raise self.logger.error(f"Model Name must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MolecularFingerprint', \
                                               'MPNN', 'DMPNN', 'CMPNN', 'MAT', 'BiLSTM', 'BiLSTM-Attention']")

        if len(validset_indices) == 0:
            valid_loader = None
        
        # print('train, valid:', len(train_loader), len(valid_loader))
        
        return train_loader, valid_loader, scaler