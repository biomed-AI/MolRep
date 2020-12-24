# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Jaeger S et al "Mol2vec: unsupervised machine learning approach with chemical intuition." -> https://github.com/samoturk/mol2vec
"""


import numpy as np
import pandas as pd
import sys, os

from rdkit import Chem
from gensim.models import word2vec

from MolRep.Models.unsupervised_based.mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec

class Mol2VecEmbeddings():
    def __init__(self, dataset_path, mol2vec_model_path, features_dir=None,
                 use_data_saving=True, configs=None, dataset_configs=None, logger=None):
        """
        Args:
            dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                                the first one contains SMILES strings of the compounds,
                                the second one contains labels.
            mol2vec_model_path (str): A path to the mol2vec pretrained-model.
            features_dir (str): A path to save processed features.
            use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                    is present, the features will be saved after calculations. Defaults to True.
        """

        self.model_name = configs.model_name
        self.dataset_name = configs.dataset_name
        self.dataset_path = Path(dataset_path) / Path(dataset_configs["path"])
        self.features_dir = Path(dataset_path).parent if features_dir is None else Path(features_dir)
        self.configs = configs

        self.model = word2vec.Word2Vec.load(mol2vec_model_path)
        self.use_data_saving = use_data_saving
        self.logger = logger

        self.task_type = dataset_configs["task_type"]
        self.multi_class = self.task_type == 'Multiclass-Classification'
        self.multiclass_num_classes = dataset_configs["multiclass_num_classes"] if self.task_type == 'Multi-Classification' else None

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
            raw_df = pd.read_csv(str(input_file) + '.csv')
            new = raw_df.mol_id.str.split('_', n = 1, expand=True)
            raw_df['mol_id'] = new[1]
            raw_df.set_index('mol_id')
        return pd.concat([mol_df, raw_df], axis=1, join='inner').reset_index(drop=True)

    def process(self):
        """
        Load and featurize data stored in a CSV file.
        """
        output_path = self.output_dir / f"{self.model_name}.pt"
        if os.path.exists(output_path):
            self.logger.info(f"Processed features existed.")
            self.logger.info(f"Loading features stored at '{output_path}'")

            x_all, y_all = pickle.load(open(output_path, "rb"))
            self.configs.dim_features = x_all.shape[1]

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            x_all, y_all = self.load_data_from_smiles(data_x, data_y)
            self.configs.dim_features = x_all.shape[1]

            if self.use_data_saving and not os.path.exists(output_path):
                self.logger.info(f"Saving features at '{output_path}'")
                pickle.dump((x_all, y_all), open(output_path, "wb"))

    def load_data_from_smiles(self, x_smiles, labels):
        sentence_all, y_all = [], []
        for smiles, label in zip(x_smiles, labels):
            try:
                mol = MolFromSmiles(smiles)
                sentence = MolSentence(mol2alt_sentence(smiles, 1))
                sentence_all.append(sentence)
                y_all.append(label)
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a RDKit Mol .\nREASON: {}'.format(smiles, e))

        x_all = [DfVec(x) for x in sentences2vec(sentence_all, self.model, unseen='UNK')]
        x_all, y_all = np.array([x.vec for x in x_all]), np.array(y_all)
        return x_all, y_all