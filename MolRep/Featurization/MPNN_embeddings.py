# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Yang et al "Analyzing Learned Molecular Representations for Property Prediction" & "A Deep Learning Approach to Antibiotic Discovery" -> https://github.com/chemprop/chemprop
Song et al "Communicative Representation Learning on Attributed Molecular Graphs" -> https://github.com/SY575/CMPNN
"""

import os
import math
import pickle
import random

from rdkit import Chem
from pathlib import Path
from typing import Callable, List, Union, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from argparse import Namespace

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from torch.utils.data.dataset import Dataset

Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]

FEATURES_GENERATOR_REGISTRY = {}


class MPNNEmbeddings:
    def __init__(self, data_df, model_name, features_path, dataset_config,
                 additional_data=None,
                 features_generator=None, use_data_saving=True, atom_descriptors=None):
        
        self.model_name = model_name
        self.whole_data_df = data_df
        self.features_path = features_path
        self.dataset_config = dataset_config
        self.dataset_name = self.dataset_config["name"]
        self.additional_data = additional_data

        self.features_generator = features_generator
        self.use_data_saving = use_data_saving
        self.atom_descriptors = atom_descriptors

        self.smiles_col = dataset_config["smiles_column"]
        self.target_cols = dataset_config["target_columns"]

    @property
    def dim_features(self):
        return self._dim_features

    @property
    def dim_edge_features(self):
        return None

    @property
    def max_num_nodes(self):
        return 200

    def process(self):
        """
        Load and featurize data stored in a CSV file.
        """

        features_path = self.features_path
        if self.use_data_saving and os.path.exists(features_path):
            # smiles_all, x_all, y_all = pickle.load(open(features_path, 'rb'))
            dataset = torch.load(features_path)
            self._dim_features = len(dataset["x_all"][0][0]) if dataset["x_all"][0][0] is not None else 0

        else:
            data_x = self.whole_data_df.loc[:,self.smiles_col].values
            data_y = self.whole_data_df.loc[:,self.target_cols].values

            smiles_all, x_all, y_all = self.load_data_from_smiles(data_x, data_y)

            self._dim_features = len(x_all[0][0]) if x_all[0][0] is not None else 0
            dataset = {
                "x_all": x_all,
                "y_all": y_all,
                "smiles_all": smiles_all,
            }
            # pickle.dump((smiles_all, x_all, y_all), open(features_path, "wb"))
            torch.save(dataset, features_path)

        return dataset


    def load_data_from_smiles(self, x_smiles, labels):
        """
         Load and featurize data from lists of SMILES strings and labels.
        Args:
            x_smiles (list[str]): A list of SMILES strings.
            labels (list[float]): A list of the corresponding labels.
        Returns:
            A tuple (SMILES, X, y) in which SMILES is a list of smiles string, X is a list of SMILES features,
            and y is a list of the corresponding labels.
        """
        smiles_all, x_all, y_all = [], [], []
        for smiles, label in zip(x_smiles, labels):
            try:
                mol = Chem.MolFromSmiles(smiles)
                features = []
                atom_features, atom_descriptors = None, None
                if self.features_generator is not None:
                    for fg in self.features_generator:
                        features_generator = get_features_generator(fg)
                        if mol is not None and mol.GetNumHeavyAtoms() > 0:
                            features.extend(features_generator(mol))
                    features = np.array(features)
                
                if len(features) == 0:
                    features = None

                # Fix nans in features
                if features is not None:
                    replace_token = 0
                    features = np.where(np.isnan(features), replace_token, features)

                if self.atom_descriptors == 'feature':
                    atom_features = self.get_atom_features(smiles)
                elif self.atom_descriptors == 'descriptor':
                    atom_descriptors = self.get_atom_descriptors(smiles)

                smiles_all.append(smiles)
                x_all.append([features, atom_features, atom_descriptors])
                if isinstance(label, np.ndarray):
                    y_all.append(label)
                else:
                    y_all.append([label])
            except ValueError as e:
                print('the SMILES ({}) can not be converted to a Molecule in RDkit.\nREASON: {}'.format(smiles, e))

        return smiles_all, x_all, np.array(y_all)


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered FeaturesGenerator by name.
    Args:
        - features_generator_name: The name of the FeaturesGenerator.

    Return: 
        The desired FeaturesGenerator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Registers a features generator.
    Args:
        - features_generator_name: The name to call the FeaturesGenerator.

    Return:
        A decorator which will add a FeaturesGenerator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_available_features_generators() -> List[str]:
    """Returns the names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())



MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048

@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.
    Args:
        - mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        - radius: Morgan fingerprint radius.
        - num_bits: Number of bits in Morgan fingerprint.

    Return:
        A 1-D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.
    Args:
        - mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        - radius: Morgan fingerprint radius.
        - num_bits: Number of bits in Morgan fingerprint.

    Return: 
        A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.
        Args:
            - mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        Return: 
            A 1D numpy array containing the RDKit 2D features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]

        return features

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.
        Args:
            - mol: A molecule (i.e. either a SMILES string or an RDKit molecule).

        Return: 
            A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features
except ImportError:
    pass

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
EXTRA_ATOM_FDIM = 0
BOND_FDIM = 14


def get_atom_fdim() -> int:
    """Gets the dimensionality of the atom feature vector."""
    return ATOM_FDIM + EXTRA_ATOM_FDIM


def set_extra_atom_fdim(extra) -> int:
    """Change the dimensionality of the atom feature vector."""
    global EXTRA_ATOM_FDIM
    EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.
    Args:
        - atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    Return: 
        The dimensionality of the bond feature vector.
    """
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    Arggs:
        - value: The value for which the encoding should be one.
        - choices: A list of possible values.

    Return: 
        A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    Args:
        - atom: An RDKit atom.
        - functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    
    Return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    Args:
        - bond: An RDKit bond.

    Return:
        A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, mol: Union[str, Chem.Mol], atom_descriptors: np.ndarray = None):
        """
        :param mol: A SMILES or an RDKit molecule.
        """
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # Get atom features
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
        if atom_descriptors is not None:
            self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in zip(self.f_atoms, atom_descriptors)]

        self.n_atoms = len(self.f_atoms)

        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:
    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.mol_graphs = mol_graphs
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_atoms.requires_grad_()
        self.f_atoms.retain_grad()

        self.f_bonds = torch.FloatTensor(f_bonds)
        self.f_bonds.requires_grad_()
        self.f_atoms.retain_grad()

        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the :class:`BatchMolGraph`.
        The returned components are, in order:
        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`
        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, :get_bond_fdim(atom_messages=atom_messages)]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.
        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(mols: Union[List[str], List[Chem.Mol]], atom_descriptors_batch: List[np.array] = None) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.
    Args:
        - mols: A list of SMILES or a list of RDKit molecules.
        - atom_descriptors_batch: A list of 2D numpy array containing additional atom descriptors to featurize the molecule
    Return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    if atom_descriptors_batch is not None:
        return BatchMolGraph([MolGraph(mol, atom_descriptors) for mol, atom_descriptors in zip(mols, atom_descriptors_batch)])
    else:
        return BatchMolGraph([MolGraph(mol) for mol in mols])



def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    Args:
        - source: A tensor of shape (num_bonds, hidden_size) containing message features.
        - index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
                 indices to select from source.
    Return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
            features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    
    target[index==0] = 0
    return target


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.
    Args:
        - activation: The name of the activation function.
    Return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')
