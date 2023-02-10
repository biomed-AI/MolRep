# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Yang et al "Analyzing Learned Molecular Representations for Property Prediction" & "A Deep Learning Approach to Antibiotic Discovery" -> https://github.com/chemprop/chemprop
Song et al "Communicative Representation Learning on Attributed Molecular Graphs" -> https://github.com/SY575/CMPNN
"""


import threading
import numpy as np
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from rdkit import Chem

from molrep.processors.mpnn_embeddings import BatchMolGraph, MolGraph
from molrep.processors.mpnn_embeddings import get_features_generator

from molrep.common.utils import worker_init
from molrep.common.registry import registry
from molrep.data.datasets.base_dataset import MoleculeDataset, MoleculeSampler


# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


def cache_graph() -> bool:
    r"""Returns whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}


def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


@registry.register_dataset("mpnn")
class MPNNDataset(MoleculeDataset):
    r"""A :class:`MoleculeDataset`, called `MPNNDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data):
        r"""
        Args:
            data: A list of :class:`MoleculeDatapoint`\ s.
        """
        super().__init__(data)
        self._batch_graph = None

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).
        return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item):
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.
        Args:
            item: An index (int) or a slice object.
        return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]
    
    @classmethod
    def construct_dataset(cls, indices, features_path):
        features = torch.load(features_path)
        smiles_all, x_all, y_all = features["smiles_all"], features["x_all"], features["y_all"]

        smiles, x, y = np.array(smiles_all)[indices], np.array(x_all)[indices], np.array(y_all)[indices]
        dataset = cls([
                MoleculeDatapoint(
                    smiles=smiles,
                    targets=targets,
                    features=x[i][0] if x[i][0] is not None else None,
                    atom_features=x[i][1] if x[i][1] is not None else None,
                    atom_descriptors=x[i][2] if x[i][2] is not None else None,
                ) for i, (smiles, targets) in enumerate(zip(smiles, y))
        ])

        return dataset

    @classmethod
    def collate_fn(cls, batch):
        batch_data = cls(batch)
        batch_data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

        smiles_batch = batch_data.batch_graph()
        target_batch = batch_data.targets()
        features_batch, atom_descriptors_batch = batch_data.features(), batch_data.atom_descriptors()

        return {
            "smiles": smiles_batch,
            "targets": target_batch,
            "features": features_batch,
            "atom_descriptors": atom_descriptors_batch,
        }
    
    def bulid_dataloader(self, config, is_train=True):

        num_workers = config.run_cfg.get("num_workers", 2)
        class_balance = config.run_cfg.get("class_balance", False)
        shuffle = (is_train == True)
        seed = config.run_cfg.get("seed", 42)

        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._data,
            class_balance=class_balance,
            shuffle=shuffle,
            seed=seed
        )

        return DataLoader(
                        dataset=self._data,
                        batch_size=config.run_cfg.batch_size,
                        sampler=self._sampler,
                        num_workers=num_workers,
                        collate_fn=self.collate_fn,
                        timeout=self._timeout,
                        worker_init_fn=worker_init,
        )

    def batch_graph(self) -> BatchMolGraph:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.
        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.
        :return: A :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the molecules.
        """
        if self._batch_graph is None:
            mol_graphs = []
            for d in self._data:
                if d.smiles in SMILES_TO_GRAPH:
                    mol_graph = SMILES_TO_GRAPH[d.smiles]
                else:
                    mol_graph = MolGraph(d.mol, d.atom_features)
                    if cache_graph():
                        SMILES_TO_GRAPH[d.smiles] = mol_graph
                mol_graphs.append(mol_graph)

            self._batch_graph = BatchMolGraph(mol_graphs)

        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).
        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).
        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self._data]

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.
        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.
        :return: The size of the additional atom descriptor vector.
        """
        return len(self._data[0].atom_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        """
        Returns the size of custom additional atom features vector associated with the molecules.
        :return: The size of the additional atom feature vector.
        """
        return len(self._data[0].atom_features[0]) \
            if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def reset_features_and_targets(self) -> None:
        """Resets the features and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: str,
                 targets: List[Optional[float]] = None,
                 features: np.ndarray = None,
                 atom_features: np.ndarray = None,
                 atom_descriptors: np.ndarray = None):
        """
        Args:
            smiles: The SMILES string for the molecule.
            targets: A list of targets for the molecule (contains None for unknown target values).
            features: A numpy array containing additional features (e.g., Morgan fingerprint).
            atom_features: A list of atom features to use.
            atom_descriptors: A list of atom descriptorss to use.
        """

        self.smiles = smiles
        self.targets = targets
        self.features = features
        self.atom_descriptors = atom_descriptors
        self.atom_features = atom_features

        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Fix nans in atom_descriptors
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors)

        # Fix nans in atom_features
        if self.atom_features is not None:
            self.atom_features = np.where(np.isnan(self.atom_features), replace_token, self.atom_features)

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features, self.raw_targets = self.features, self.targets

    @property
    def mol(self) -> Chem.Mol:
        """Gets the corresponding RDKit molecule for this molecule's SMILES."""
        mol = SMILES_TO_MOL.get(self.smiles, Chem.MolFromSmiles(self.smiles))

        if cache_mol():
            SMILES_TO_MOL[self.smiles] = mol

        return mol

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.
        Args:
            features: A 1D numpy array of features for the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.
        return:
            The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.
        Args:
                targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features and targets to their raw values."""
        self.features, self.targets = self.raw_features, self.raw_targets