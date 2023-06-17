#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023, Sun Yat-sen Univeristy.
All rights reserved.

@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
'''


import sklearn
import numpy as np

import torch
from torch_geometric.data import Batch

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib as mpl
GREEN_COL = mpl.colors.to_rgb("#1BBC9B")
RED_COL = mpl.colors.to_rgb("#F06060")


def explainer_data(data, explainer, model, device=None, **kwargs):
    device = model.device if device is None else device
    atom_weights, bond_weights = [], []
    for batch_data in data:
        for k, v in batch_data.items():
            if type(v) == torch.Tensor or issubclass(type(v), Batch):
                batch_data[k] = v.to(device, non_blocking=True)

        node_scores, edge_scores = explainer.explain(batch_data, model, **kwargs)

        if node_scores is not None:
            atom_weights.extend(node_scores)
        
        if edge_scores is not None:
            bond_weights.extend(edge_scores)

    return atom_weights, bond_weights


def preprocessing_attributions(smiles_list, atom_importance, bond_importance, normalizer=None):
    att_probs = []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        atom_imp = atom_importance[idx].cpu().detach().numpy()

        if bond_importance is not None and len(bond_importance) > 0:
            bond_imp = bond_importance[idx]

            bond_idx = []
            for bond in mol.GetBonds():
                bond_idx.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

            for (atom_i_idx, atom_j_idx), b_imp in zip(bond_idx, bond_imp):
                atom_imp[atom_i_idx] += b_imp / 2
                atom_imp[atom_j_idx] += b_imp / 2
        att_probs.append(atom_imp)
    att_probs = [att[:, -1] if att_probs[0].ndim > 1 else att for att in att_probs]

    if normalizer is not None:
        att_probs = normalize_attributions(att_probs, normalizer=normalizer)
    return att_probs


def normalize_attributions(att_list, positive=False, normalizer='MinMaxScaler'):
    """Normalize all nodes to 0 to 1 range via quantiles."""
    all_values = np.concatenate(att_list)
    all_values = all_values[all_values > 0] if positive else all_values

    if normalizer == 'QuantileTransformer':
        normalizer = sklearn.preprocessing.QuantileTransformer()
    elif normalizer == 'MaxAbsScaler':
        normalizer = sklearn.preprocessing.MaxAbsScaler()
    else:
        normalizer = sklearn.preprocessing.MinMaxScaler()
    normalizer.fit(all_values.reshape(-1, 1))
    
    new_att = []
    for att in att_list:
        normed_nodes = normalizer.transform(att.reshape(-1, 1)).ravel()
        new_att.append(normed_nodes)
    return new_att


def visualize(smiles_list, atom_importance, **kwargs):
    svg_list = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        cp = Chem.Mol(mol)
        
        atom_imp = atom_importance[i]

        highlightAtomColors, cp = determine_atom_col(cp, atom_imp, eps=kwargs.get('threshold', 1e-4), use_negative=kwargs.get('use_negative', False), set_weights=kwargs.get('set_weights', False))
        highlightAtoms = list(highlightAtomColors.keys())

        highlightBondColors = determine_bond_col(highlightAtomColors, mol)
        highlightBonds = list(highlightBondColors.keys())

        rdDepictor.Compute2DCoords(cp, canonOrient=True)
        drawer = rdMolDraw2D.MolDraw2DSVG(kwargs.get('img_width', 400), kwargs.get('img_height', 200))
        if kwargs.get('drawAtomIndices', False):
            drawer.drawOptions().addAtomIndices = True
        drawer.drawOptions().useBWAtomPalette()
        drawer.DrawMolecule(
            cp,
            highlightAtoms=highlightAtoms,
            highlightAtomColors=highlightAtomColors,
            highlightBonds=highlightBonds,
            highlightBondColors=highlightBondColors,
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg_list.append(svg.replace('svg:',''))
    return svg_list


def determine_atom_col(cp, atom_importance, eps=1e-5, use_negative=True, set_weights=False):
    """ Colors atoms with positive and negative contributions
    as green and red respectively, using an `eps` absolute
    threshold.

    Parameters
    ----------
    mol : rdkit mol
    atom_importance : np.ndarray
        importances given to each atom
    bond_importance : np.ndarray
        importances given to each bond
    version : int, optional
        1. does not consider bond importance
        2. bond importance is taken into account, but fixed
        3. bond importance is treated the same as atom importance, by default 2
    eps : float, optional
        threshold value for visualization - absolute importances below `eps`
        will not be colored, by default 1e-5

    Returns
    -------
    dict
        atom indexes with their assigned color
    """
    atom_col = {}

    for idx, v in enumerate(atom_importance):
        if v > eps:
            atom_col[idx] = GREEN_COL
        if use_negative and v < -eps:
            atom_col[idx] = RED_COL
        if set_weights:
            cp.GetAtomWithIdx(idx).SetProp("atomNote","%.3f"%(v))
    return atom_col, cp


def determine_bond_col(atom_col, mol):
    """Colors bonds depending on whether the atoms involved
    share the same color.

    Parameters
    ----------
    atom_col : np.ndarray
        coloring assigned to each atom index
    mol : rdkit mol

    Returns
    -------
    dict
        bond indexes with assigned color
    """
    bond_col = {}

    for idx, bond in enumerate(mol.GetBonds()):
        atom_i_idx, atom_j_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if atom_i_idx in atom_col and atom_j_idx in atom_col:
            if atom_col[atom_i_idx] == atom_col[atom_j_idx]:
                bond_col[idx] = atom_col[atom_i_idx]
    return bond_col
