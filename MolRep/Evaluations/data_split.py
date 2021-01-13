

from collections import defaultdict
from random import Random
from typing import Dict, List, Set, Tuple, Union
import warnings

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np


def specific_split(all_indices, cell_type, test_size=0, random_state=0):
    train_idxs, test_idxs = [], []
    for i, c_t in enumerate(cell_type):
        if c_t == '143B':
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    return np.array(train_idxs), np.array(test_idxs)

def scaffold_split(data,
                   test_size,
                   balanced = False,
                   random_state = 0):
    """
    Splits a :class:`~chemprop.data.MoleculeDataset` by scaffold so that no molecules sharing a scaffold are in different splits.
    :param data
    :param test_size: the proportions of data in the test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param random_state: Random seed for shuffling when doing balanced splitting.
    :return: A tuple containing the train, validation, and test splits of the data.
    """

    sizes = [1-test_size, 0, test_size]
    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)

    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0


    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data, use_indices=True)

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

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
    
    return train, val, test


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds