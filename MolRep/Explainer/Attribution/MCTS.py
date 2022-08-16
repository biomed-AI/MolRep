
import abc
import torch
from typing import Any, Callable, List, MutableMapping, Optional, Text, Tuple


import math
from functools import partial
from multiprocessing import Pool

import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs

import os
import pickle

from MolRep.Explainer.Attribution.utils.fuseprop import find_clusters, extract_subgraph
from MolRep.Explainer.Attribution.utils.fuseprop.chemprop_utils import *
from MolRep.Explainer.Attribution.utils.gnn_utils import *

MIN_ATOMS = 3
C_PUCT = 10

class MCTSNode():

    def __init__(self, smiles, atoms, W=0, N=0, P=0):
        self.smiles = smiles
        self.atoms = set(atoms)
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)

class MCTS:

    def __init__(self, name: Optional[Text] = None, 
                 rollout=20, c_puct=10, max_atoms=15, min_atoms=3, prop_delta=0.3, ncand=1, ncpu=2):
        self.name = name or self.__class__.__name__
        self.rollout = rollout
        self.c_puct = c_puct
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.prop_delta = prop_delta
        self.ncand = ncand
        self.ncpu = ncpu

    def attribute(self, data, model, model_name, scaler=None):
        # smiles, mol_batch, features_batch, atom_descriptors_batch = data
        # data = (mol_batch, features_batch, atom_descriptors_batch)
        smiles = data.smiles[0]

        output = model(data)
        if not isinstance(output, tuple):
            output = (output,)

        results = mcts_search(smiles, model, model_name, scaler,
                              rollout=self.rollout, max_atoms=self.max_atoms,
                              prop_delta=self.prop_delta, ncpu=self.ncpu)
        
        node_weights = self.get_attribution_results(smiles, results)
        return node_weights, None, output

    def get_attribution_results(self, smiles, results):

        mol = Chem.MolFromSmiles(smiles)
        node_weights = np.zeros(len(mol.GetAtoms()))

        orig_smiles, rationales = results
        rationales = sorted(rationales, key=lambda x:len(x.atoms))
        for x in rationales[:self.ncand]:
            mol = Chem.MolFromSmiles(orig_smiles)
            sub = Chem.MolFromSmiles(x.smiles)
            
            subgraph_truth = mol.GetSubstructMatch(sub)
            node_weights[list(subgraph_truth)] = 1

        return node_weights


def mcts_rollout(node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function):
    print('cur_node', node.smiles)
    cur_atoms = node.atoms
    if len(cur_atoms) <= MIN_ATOMS:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        cur_cls = set( [i for i,x in enumerate(clusters) if x <= cur_atoms] )
        for i in cur_cls:
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                #p rint('new_smiles', node.smiles, '->', new_smiles)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles] # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0: return node.P  # cannot find leaves

        scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            if np.array(score).ndim == 1:
                child.P = score
            else:
                child.P = 1 - score[0][0]
        
    sum_count = sum([c.N for c in node.children])
    selected_node = max(node.children, key=lambda x : x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function)
    selected_node.W += v
    selected_node.N += 1
    return v

def mcts(smiles, scoring_function, n_rollout, max_atoms, prop_delta):
    mol = Chem.MolFromSmiles(smiles)
    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i,cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - set([i])
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])
    
    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())) ) 
    state_map = {smiles : root}
    for _ in range(n_rollout):
        mcts_rollout(root, state_map, smiles, clusters, atom_cls, nei_cls, scoring_function)

    rationales = [node for _,node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]
    return smiles, rationales


def mcts_search(data, model, model_name, scaler=None,
                rollout=20, max_atoms=15, prop_delta=0.3, ncpu=4):

    scoring_function = get_scoring_function(model, model_name, scaler)
    # work_func = partial(mcts, scoring_function=scoring_function, 
    #                           n_rollout=rollout, 
    #                           max_atoms=max_atoms, 
    #                           prop_delta=prop_delta)

    # pool = Pool(ncpu)
    # results = pool.map(work_func, data)
    results = mcts(smiles=data, scoring_function=scoring_function, n_rollout=rollout, max_atoms=max_atoms, prop_delta=prop_delta)
    return results


class chemprop_model():

    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

    def __call__(self, smiles, batch_size=1):
        test_data = get_data_from_smiles(smiles=smiles)
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        model_preds = predict(
            model=self.model,
            data=test_data,
            batch_size=batch_size,
            scaler=self.scaler
        )

        # Put zero for invalid smiles
        full_preds = [0.0] * len(full_data)
        for i, si in enumerate(valid_indices):
            full_preds[si] = model_preds[i]

        return np.array(full_preds, dtype=np.float32)


class gnn_model:

    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler


    def __call__(self, smiles, batch_size=500):
        test_data, valid_indices = get_gnn_data_from_smiles(smiles=smiles)
        
        model_preds = gnn_predict(
            model=self.model,
            test_data=test_data,
            scaler=self.scaler,
            device='cuda'
        )

        # Put zero for invalid smiles
        full_preds = [0.0] * len(smiles)
        for i, si in enumerate(valid_indices):
            full_preds[si] = model_preds[i]

        return np.array(full_preds, dtype=np.float32)


def get_scoring_function(model, model_name, scaler):
    """Function that initializes and returns a scoring function by name"""
    if model_name in ['DMPNN', 'CMPNN']:
        return chemprop_model(model=model, scaler=scaler)
    if model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'GAT', 'PyGCMPNN']:
        return gnn_model(model=model, scaler=scaler)
