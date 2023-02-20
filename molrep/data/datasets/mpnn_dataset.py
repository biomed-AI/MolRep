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
from rdkit import Chem

from molrep.processors.features import BatchMolGraph, MolGraph

from molrep.common.registry import registry
from molrep.data.datasets.base_dataset import MoleculeDataset


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
    def collate_fn(cls, batch, **kwargs):
        atom_messages = kwargs["atom_messages"] if "atom_messages" in kwargs.keys() else False
        batch_data = cls(batch)
        smiles_batch = batch_data.batch_graph(atom_messages)  # Forces computation and caching of the BatchMolGraph for the molecules
        target_batch = batch_data.targets()
        features_batch, atom_descriptors_batch = batch_data.features(), batch_data.atom_descriptors()

        return {
            "smiles": smiles_batch,
            "targets": target_batch,
            "features": features_batch,
            "atom_descriptors": atom_descriptors_batch,
        }

    def batch_graph(self, atom_messages=False) -> BatchMolGraph:
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

            self._batch_graph = BatchMolGraph(mol_graphs, atom_messages=atom_messages)
        return self._batch_graph

    def smiles(self) -> List[str]:
        return [d.smiles for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]
    
    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            target = torch.FloatTensor([targets[i]])
            self._data[i].targets = target.unsqueeze(1) if target.ndim == 1 else target

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