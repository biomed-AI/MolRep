# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Song et al "Communicative Representation Learning on Attributed Molecular Graphs" -> https://github.com/SY575/CMPNN
"""

from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from MolRep.Featurization.MPNN_embeddings import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from MolRep.Featurization.MPNN_embeddings import index_select_ND, get_activation_function

import math
import torch.nn.functional as F


class CMPNN(nn.Module):
    """A CMPNN is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, max_num_nodes=200):
        """
        Initializes the CMPNN.
        :param classification: Whether the model is a classification model.
        """
        super(CMPNN, self).__init__()

        self.dim_features = dim_features
        self.dim_target = dim_target
        self.max_num_nodes = max_num_nodes

        self.model_configs = model_configs

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
        assert not (self.classification and self.regression and self.multiclass)

        self.create_encoder()
        self.create_ffn()
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
        :param model_configs: Arguments.
        """
        self.encoder = MPN(self.model_configs, max_num_nodes=self.max_num_nodes)

    def create_ffn(self):
        """
        Creates the feed-forward network for the model.
        :param args: Arguments.
        """
        args = self.model_configs

        self.multiclass = self.task_type == 'Multi-Classification'
        if self.multiclass:
            self.num_classes = self.multiclass_num_classes
        if args['features_only']:
            first_linear_dim = self.dim_features
        else:
            first_linear_dim = int(args['dim_embedding']) * 1
            first_linear_dim += self.dim_features

        dropout = nn.Dropout(args['dropout'])
        activation = get_activation_function(args['activation'])

        # Create FFN layers
        if args['ffn_num_layers'] == 1:
            self.last_linear = nn.Linear(first_linear_dim, self.dim_target)
            ffn = [
                dropout,
                self.last_linear
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args['dim_embedding'])
            ]
            for _ in range(args['ffn_num_layers'] - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args['dim_embedding'], args['dim_embedding']),
                ])

            self.last_linear = nn.Linear(args['dim_embedding'], self.dim_target)
            ffn.extend([
                activation,
                dropout,
                self.last_linear,
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def featurize(self, data):
        batch, features_batch, _ = data
        output = self.encoder.featurize(batch, features_batch)
        return output

    def forward(self, data):
        """
        Runs the CMPNN on input.
        :param input: Input.
        :return: The output of the CMPNN.
        """
        batch, features_batch, _ = data
        output = self.ffn(self.encoder(batch, features_batch))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output

    def get_intermediate_activations_gradients(self, data):
        batch, features_batch, _ = data
        output = self.ffn(self.encoder(batch, features_batch))

        conv_acts, conv_grads = self.encoder.encoder.get_intermediate_activations_gradients(output)
        conv_grads = [conv[1:, :] for conv in conv_grads]
        return conv_acts, conv_grads

    def get_gap_activations(self, data):
        batch, features_batch, _ = data
        output = self.ffn(self.encoder(batch, features_batch))
        
        conv_acts, _ = self.encoder.encoder.get_intermediate_activations_gradients(output)
        return conv_acts[-1], None

    def get_prediction_weights(self):
        w = self.last_linear.weight.t()
        return w[:,0]

    def get_gradients(self, data):
        batch, features_batch, _ = data
        output = self.ffn(self.encoder(batch, features_batch))

        self.encoder.encoder.get_gradients(output)

        atom_grads = self.encoder.encoder.f_atoms.grad
        bond_grads = self.encoder.encoder.f_bonds.grad
        return self.encoder.encoder.f_atoms[1:, :], atom_grads[1:, :], None, None


class MPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int, max_num_nodes: int = 200):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args['dim_embedding']
        self.bias = args['bias']
        self.depth = args['depth']
        self.dropout = args['dropout']
        self.layers_per_message = 1
        self.undirected = args['undirected']
        self.atom_messages = args['atom_messages']
        self.features_only = args['features_only']
        self.max_num_nodes = max_num_nodes
        self.args = args
        self.f_atoms, self.f_bonds, self.node_emb = None, None, None

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args['activation'])

        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
        
        self.W_o = nn.Linear(
                (self.hidden_size)*2,
                self.hidden_size)
        
        self.gru = BatchGRU(self.hidden_size)
        
        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)

    def get_gradients(self, output):
        self.f_atoms.retain_grad()
        self.f_bonds.retain_grad()
        output.backward(torch.ones_like(output))
        return self.f_atoms.grad, self.f_bonds.grad

    def get_intermediate_activations_gradients(self, output):
        output.backward(torch.ones_like(output))
        conv_grads = [conv_g.grad for conv_g in self.conv_grads]
        return self.conv_acts, self.conv_grads

    def activations_hook(self, grad):
        self.conv_grads.append(grad)

    def featurize(self, mol_graph: BatchMolGraph, features_batch=None):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        if self.args['device'] == 'cuda' or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda())

        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)

        # Message passing
        for depth in range(self.depth - 1):

            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message
            
            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
            
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)
        
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        # Readout
        feat = torch.zeros((len(a_scope), self.max_num_nodes, self.hidden_size)).to(atom_hiddens.device)
        mask = torch.ones((len(a_scope), self.max_num_nodes)).to(atom_hiddens.device)

        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            if a_size <= self.max_num_nodes:
                feat[i, :a_size] = atom_hiddens.narrow(0, a_start, a_size)
                mask[i, a_size:] = 0
            else:
                feat[i] = atom_hiddens.narrow(0, a_start, self.max_num_nodes)
        
        return feat, mask


    def forward(self, mol_graph: BatchMolGraph, features_batch=None) -> torch.FloatTensor:

        # f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        if self.args['device'] == 'cuda' or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda())

        f_atoms.requires_grad_()
        f_atoms.retain_grad()
        f_bonds.requires_grad_()
        f_bonds.retain_grad()
        self.f_atoms, self.f_bonds = f_atoms, f_bonds

        self.conv_acts = []
        self.conv_grads = []

        # Input
        with torch.enable_grad():
            input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
            input_atom = self.act_func(input_atom)
            message_atom = input_atom.clone()
            
            input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
            message_bond = self.act_func(input_bond)
            input_bond = self.act_func(input_bond)

        # Message passing
        for depth in range(self.depth - 1):
            message_atom.register_hook(self.activations_hook)
            self.conv_acts.append(message_atom[1:, :])

            with torch.enable_grad():
                agg_message = index_select_ND(message_bond, a2b)
                agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
                message_atom = message_atom + agg_message
                
                # directed graph
                rev_message = message_bond[b2revb]  # num_bonds x hidden
                message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
                
                message_bond = self._modules[f'W_h_{depth}'](message_bond)
                message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)
        
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = torch.stack(mol_vecs, dim=0)
        
        return mol_vecs  # B x H


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message


class MPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 max_num_nodes: int = 200):
        super(MPN, self).__init__()
        self.args = args
        self.atom_descriptors = None
        
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args['atom_messages'])
        # self.atom_fdim = atom_fdim or get_atom_fdim(args)
        # self.bond_fdim = bond_fdim or get_bond_fdim(args) + \
        #                     (not args.atom_messages) * self.atom_fdim # * 2
        
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim, max_num_nodes)

    def featurize(self, batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:

        return self.encoder.featurize(batch, features_batch)

    def forward(self, batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:

        if type(batch) != BatchMolGraph:
            if self.atom_descriptors == 'feature':
                batch = mol2graph(batch)
            else:
                batch = mol2graph(batch)
    
        output = self.encoder.forward(batch, features_batch)

        return output