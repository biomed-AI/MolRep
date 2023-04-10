

import os
import json
import numpy as np
import pandas as pd

from pathlib import Path

from molrep.common.registry import registry
from molrep.dataset.builders.base_builder import BaseDatasetBuilder
from molrep.processors.pretraining_embeddings import _load_chembl_with_labels_dataset
from molrep.processors.pretraining_transform import NegativeEdge, MaskAtom, ExtractSubstructureContextPair

from ogb.graphproppred import PygGraphPropPredDataset

@registry.register_builder("molecular_pretraining")
class PretrainingBuilder(BaseDatasetBuilder):
    
    pretrain_model_dataset_mapping = {
        "attr_masking": "pretraining", "edge_pred": "pretraining", 
        "context_pred": "pretraining",
    }

    pretrain_model_transform_mapping = {
        "attr_masking": MaskAtom, "edge_pred": NegativeEdge, 
        "context_pred": ExtractSubstructureContextPair,
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def build_datasets(self):
        self._download_and_load_data()
        datasets, _ = self.build()
        return datasets, None

    def _download_and_load_data(self):

        self.model_name = self.config.model_cfg.name
        self.dataset_name = self.config.datasets_cfg["name"]

        if self.dataset_name == 'zinc_standard_agent':
            dataset_path = Path(os.path.join(self.cache_root, self.dataset_config.storage[0]))
            if dataset_path.suffix == '.csv':
                self.whole_data_df = pd.read_csv(dataset_path)

            self.smiles_col = self.config.datasets_cfg.smiles_column
            self.molid_col = self.config.datasets_cfg.id_column

            self.smiles_list = list(self.whole_data_df[self.smiles_col])
            self.processor_kwargs = {
                "zinc_ids": list(self.whole_data_df[self.molid_col[0]]),
            }

        elif self.dataset_name == 'chembl_filtered':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.cache_root, self.dataset_config.storage[0]))
            self.smiles_list = smiles_list
            self.processor_kwargs = {
                "rdkit_mol_objs": rdkit_mol_objs,
                "folds": folds, "labels": labels,
            }

        else:
            raise NotImplementedError()

    def build(self):

        processor_cls = registry.get_processor_class("pretraining")(self.config, self.smiles_list, **self.processor_kwargs)
        processor_cls.process()

        self.config.datasets_cfg.dim_features = processor_cls.dim_features
        datasets, _ = self.construct_datasets()
        return datasets, None

    def construct_datasets(self):
        self.transform = self.pretrain_model_transform_mapping[self.model_name](self.config)
        self.features_dir = Path(registry.get_path("features_root"))
        self.features_path = self.features_dir / f"{self.dataset_name}" / f"{self.model_name}.pt"

        datasets = {
            "train": None,
        }

        dataset_cls = registry.get_dataset_class(self.pretrain_model_dataset_mapping[self.model_name])
        train_dataset = dataset_cls.construct_dataset(self.features_path, transform=self.transform)

        datasets["train"] = train_dataset
        return datasets, None


