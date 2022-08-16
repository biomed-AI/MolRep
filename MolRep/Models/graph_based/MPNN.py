# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Yang et al "Analyzing Learned Molecular Representations for Property Prediction" & "A Deep Learning Approach to Antibiotic Discovery" -> https://github.com/chemprop/chemprop
"""

from argparse import Namespace
from typing import List, Union

from rdkit import Chem

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from MolRep.Featurization.MPNN_embeddings import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from MolRep.Featurization.MPNN_embeddings import index_select_ND, get_activation_function


class MPNN(nn.Module):
    """A :class:`MPNN` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs):
        """

        """
        super(MPNN, self).__init__()

        self.model_configs = model_configs
        self.dim_features = dim_features
        self.dim_target = dim_target

        self.task_type = dataset_configs["task_type"]
        self.multiclass_num_classes = dataset_configs["multiclass_num_classes"] if self.task_type == 'Multi-Classification' else None

        self.classification = self.task_type == 'Classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.multiclass = self.task_type == 'Multi-Classification'
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.regression = self.task_type == 'Regression'
        if self.regression:
            self.relu = nn.ReLU()

        self.dropout = nn.Dropout(model_configs['dropout'])
        if model_configs['features_only']:
            self.first_linear_dim = self.dim_features
        else:
            self.first_linear_dim = int(model_configs['dim_embedding']) * 1
            self.first_linear_dim += self.dim_features
        ffn = [
                self.dropout,
                nn.Linear(self.first_linear_dim, self.dim_target)
            ]
        self.ffn = nn.Sequential(*ffn)

        self.create_encoder()
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of a model in place.
        """
        for param in self.encoder.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        for param in self.ffn.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def create_encoder(self):
        """
        Creates the message passing encoder for the model.
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(self.model_configs)

    def forward(self, data) -> torch.FloatTensor:
        """
        Runs the :class:`MPNN` on input.
        :param batch Union[List[str], List[Chem.Mol], BatchMolGraph]: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch List[np.ndarray] = None: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch List[np.ndarray] = None: A list of numpy arrays containing additional atom descriptors.
        :return: The output of the :class:`MPNN`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        batch, features_batch, atom_descriptors_batch = data

        output = self.ffn(self.encoder(batch, features_batch, atom_descriptors_batch))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            # if not self.training:
            output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args, atom_fdim, bond_fdim):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.dim_embedding
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout_layer(atom_hiddens)                             # num_atoms x (hidden + descriptor size)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation=='mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation=='sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation=='norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args.atom_messages)

        self.atom_descriptors = args.atom_descriptors

        self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.
        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch) != BatchMolGraph:
            if self.atom_descriptors == 'feature':
                batch = mol2graph(batch, atom_descriptors_batch)
            else:
                batch = mol2graph(batch)

        if self.atom_descriptors == 'descriptor':
            output = self.encoder.forward(batch, features_batch, atom_descriptors_batch)
        else:
            output = self.encoder.forward(batch, features_batch)

        return output