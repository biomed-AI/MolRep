# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
Maziarka, Łukasz, et al. "Molecule Attention Transformer."  -> https://github.com/ardigen/MAT
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_


class CoMPT(nn.Module):
    """docstring for CoMPT"""
    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, max_num_nodes=200):
        super(CoMPT, self).__init__()
        self.d_atom = dim_features
        self.d_edge = 13

        c = copy.deepcopy

        attn = MultiHeadedAttention(model_configs['h'], model_configs['d_model'], model_configs['leaky_relu_slope'], model_configs['dropout'],
                                    model_configs['attenuation_lambda'], model_configs['distance_matrix_kernel'])
        ff = PositionwiseFeedForward(model_configs['d_model'], model_configs['n_dense'], model_configs['dropout'], 
                                     model_configs['leaky_relu_slope'], model_configs['dense_output_nonlinearity'])
        self.encoder = Encoder(
            EncoderLayer(model_configs['d_model'], c(attn), c(ff), model_configs['dropout'], model_configs['scale_norm']),
            model_configs['N'],
            model_configs['scale_norm']
        )
        self.node_embed = Node_Embeddings(self.d_atom, model_configs['d_model'], model_configs['dropout'])
        self.edge_embed = Edge_Embeddings(self.d_edge, model_configs['d_model'], model_configs['dropout'])
        self.post_embed = Position_Encoding(100, model_configs['d_model'], model_configs['dropout'])
        self.generator = Generator(model_configs['d_model'], dim_target, model_configs['n_generator_layers'], 
                                   model_configs['leaky_relu_slope'], model_configs['dropout'], model_configs['scale_norm'], model_configs['aggregation_type'])

        self.model = GraphTransformer(self.encoder, self.node_embed, self.edge_embed, self.post_embed, self.generator)

        self.init_type = model_configs['init_type']

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

        self._init_weights()

    def _init_weights(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                if self.init_type == 'uniform':
                    nn.init.xavier_uniform_(p)
                elif self.init_type == 'normal':
                    nn.init.xavier_normal_(p)
                elif self.init_type == 'small_normal_init':
                    xavier_normal_small_init_(p)
                elif self.init_type == 'small_uniform_init':
                    xavier_uniform_small_init_(p)
    
    def forward(self, data):
        node_features, edge_features, adjacency_matrix = data
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

        out = self.model(node_features, batch_mask, adjacency_matrix, edge_features)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            out = self.sigmoid(out)
        if self.multiclass:
            out = out.reshape((out.size(0), -1, self.multiclass_num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                out = self.multiclass_softmax(out) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        return out



class GraphTransformer(nn.Module):
    def __init__(self, encoder, node_embed, edge_embed, pos_embed, generator):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.pos_embed = pos_embed
        self.generator = generator

    def forward(self, node_features, node_mask, adj_matrix, edge_features):
        """Take in and process masked src and target sequences."""
        # return self.predict(self.encode(src, src_mask, adj_matrix, edges_att), src_mask)
        return self.predict(self.encode(node_features, edge_features, adj_matrix, node_mask), node_mask)

    def encode(self, node_features, edge_features, adj_matrix, node_mask):  # (batch, max_length, d_atom+1)
        # xv.shape = (batch, max_length, d_model)
        node_initial = self.node_embed(node_features[:, :, :-1]) + self.pos_embed(node_features[:, :, -1].squeeze(-1).long())
        # evw = xv + evw for directions; evw.shape = (batch, max_length, max_length, d_model)
        edge_initial = node_initial.unsqueeze(-2) + self.edge_embed(edge_features)
        return self.encoder(node_initial, edge_initial, adj_matrix, node_mask)

    def predict(self, out, out_mask):
        return self.generator(out, out_mask)


# Embeddings


class Node_Embeddings(nn.Module):
    def __init__(self, d_atom, d_emb, dropout):
        super(Node_Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom-1, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):  # x.shape(batch, max_length, d_atom)
        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)


class Edge_Embeddings(nn.Module):
    def __init__(self, d_edge, d_emb, dropout):
        super(Edge_Embeddings, self).__init__()
        self.lut = nn.Linear(d_edge, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):  # x.shape = (batch, max_length, max_length, d_edge)
        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)


class Position_Encoding(nn.Module):
    def __init__(self, max_length, d_emb, dropout):
        super(Position_Encoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_length + 1, d_emb, padding_idx=0)

    def forward(self, x):
        return self.dropout(self.pe(x))  # (batch, max_length) -> (batch, max_length, d_emb)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def mish_function(x):
    return x * torch.tanh(F.softplus(x))


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, n_output=1, n_layers=1,
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False, aggregation_type='mean'):
        super(Generator, self).__init__()
        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            self.proj = []
            for i in range(n_layers - 1):
                self.proj.append(nn.Linear(d_model, d_model))
                self.proj.append(Mish())
                self.proj.append(ScaleNorm(d_model) if scale_norm else LayerNorm(d_model))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_model, n_output))
            self.proj = torch.nn.Sequential(*self.proj)

        self.aggregation_type = aggregation_type
        self.leaky_relu_slope = leaky_relu_slope

        if self.aggregation_type == 'gru':
            self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(2 * d_model, d_model)
            self.bias = nn.Parameter(torch.Tensor(d_model))
            self.bias.data.uniform_(-1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_model))

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask  # (batch, max_length, d_model)

        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=1)
            out_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_pooling = out_sum
        elif self.aggregation_type == 'summax':
            out_sum = torch.sum(out_masked, dim=1)
            out_max = torch.max(out_masked, dim=1)[0]
            out_pooling = out_sum * out_max
        elif self.aggregation_type == 'gru':
            # (batch, max_length, d_model)
            out_hidden = mish_function(out_masked + self.bias)
            out_hidden = torch.max(out_hidden, dim=1)[0].unsqueeze(0)  # (1, batch, d_model)
            out_hidden = out_hidden.repeat(2, 1, 1)  # (2, batch, d_model)

            cur_message, cur_hidden = self.gru(out_masked, out_hidden)  # message = (batch, max_length, 2 * d_model)
            cur_message = mish_function(self.linear(cur_message))  # (batch, max_length, d_model)

            out_sum = cur_message.sum(dim=1)  # (batch, d_model)
            mask_sum = mask.sum(dim=1)
            out_pooling = out_sum / mask_sum  # (batch, d_model)
        else:
            out_pooling = out_masked
        projected = self.proj(out_pooling)
        return projected


# Encoder


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N, scale_norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)

    def forward(self, node_hidden, edge_hidden, adj_matrix, mask):
        """Pass the input (and mask) through each layer in turn."""
        out_scores, in_scores = None, None
        for layer in self.layers:
            node_hidden, edge_hidden = layer(node_hidden, edge_hidden, adj_matrix, mask)
        return self.norm(node_hidden)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn        # MultiHeadedAttention
        self.feed_forward = feed_forward  # PositionwiseFeedForward
        # self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 2)
        self.size = size
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_hidden, edge_hidden, adj_matrix, mask):
        """Follow Figure 1 (left) for connections."""
        # x.shape = (batch, max_length, d_atom)
        node_hidden = self.dropout(self.norm(node_hidden))
        node_hidden_first, edge_hidden_temp = self.self_attn(node_hidden, node_hidden, edge_hidden, adj_matrix, mask)
        # the first residue block
        node_hidden_first = node_hidden + self.dropout(self.norm(node_hidden_first))
        node_hidden_second = self.feed_forward(node_hidden_first)
        # the second residue block
        return node_hidden + node_hidden_first + self.dropout(self.norm(node_hidden_second)), edge_hidden


# Conv 1x1 aka Positionwise feed forward

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.1, dense_output_nonlinearity='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'gelu':
            self.dense_output_nonlinearity = lambda x: F.gelu(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
        elif dense_output_nonlinearity == 'swish':
            self.dense_output_nonlinearity = lambda x: x * torch.sigmoid(x)
        elif dense_output_nonlinearity == 'mish':
            self.dense_output_nonlinearity = lambda x: x * torch.tanh(F.softplus(x))

    def forward(self, node_hidden):
        if self.N_dense == 0:
            return node_hidden

        for i in range(self.N_dense - 1):
            node_hidden = self.dropout[i](mish_function(self.linears[i](node_hidden)))

        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](node_hidden)))


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm



# Attention


def attention(query, key, value, adj_matrix, mask=None,dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # query.shape = (batch, h, max_length, d_e)
    # key.shape = (batch, h, max_length, max_length, d_e)
    # value.shape = (batch, h, max_length, d_e)
    # out_scores.shape = (batch, h, max_length, max_length)
    # in_scores.shape = (batch, h, max_length, max_length)

    d_e = query.size(-1)
    out_scores = torch.einsum('bhmd,bhmnd->bhmn', query, key) / math.sqrt(d_e)
    in_scores = torch.einsum('bhnd,bhmnd->bhnm', query, key) / math.sqrt(d_e)

    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1)
        out_scores = out_scores.masked_fill(mask == 0, -np.inf)
        in_scores = in_scores.masked_fill(mask == 0, -np.inf)

    out_attn = F.softmax(out_scores, dim=-1)
    in_attn = F.softmax(in_scores, dim=-1)
    diag_attn = torch.diag_embed(torch.diagonal(out_attn, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

    message = out_attn + in_attn - diag_attn

    # add the diffusion caused by distance
    message = message * adj_matrix.unsqueeze(1)

    if dropout is not None:
        message = dropout(message)

    # message.shape = (batch, h, max_length, max_length), value.shape = (batch, h, max_length, d_k)
    node_hidden = torch.einsum('bhmn,bhnd->bhmd', message, value)
    edge_hidden = message.unsqueeze(-1) * key

    return node_hidden, edge_hidden, message


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, leaky_relu_slope=0.1, dropout=0.1, attenuation_lambda=0.1, distance_matrix_kernel='softmax'):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # We assume d_v always equals d_k
        self.h = h

        self.attenuation_lambda = torch.nn.Parameter(torch.tensor(attenuation_lambda, requires_grad=True))

        self.linears = clones(nn.Linear(d_model, d_model), 5)  # 5 for query, key, value, node update, edge update

        self.message = None
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = nn.Dropout(p=dropout)

        if distance_matrix_kernel == 'softmax':
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        elif distance_matrix_kernel == 'exp':
            self.distance_matrix_kernel = lambda x: torch.exp(-x)

    def forward(self, query_node, value_node, key_edge, adj_matrix, mask=None):
        """Implements Figure 2"""
        mask = mask.unsqueeze(1) if mask is not None else mask
        n_batches, max_length, d_model = query_node.shape

        # 1) Prepare adjacency matrix with shape (batch, max_length, max_length)
        torch.clamp(self.attenuation_lambda, min=0, max=1)
        adj_matrix = self.attenuation_lambda * adj_matrix
        adj_matrix = adj_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
        adj_matrix = self.distance_matrix_kernel(adj_matrix)

        # 2) Do all the linear projections in batch from d_model => h x d_k
        query = self.linears[0](query_node).view(n_batches, max_length, self.h, self.d_k).transpose(1, 2)
        key = self.linears[1](key_edge).view(n_batches, max_length, max_length, self.h, self.d_k).permute(0, 3, 1, 2, 4)
        value = self.linears[2](value_node).view(n_batches, max_length, self.h, self.d_k).transpose(1, 2)

        # 3) Apply attention on all the projected vectors in batch.
        node_hidden, edge_hidden, self.message = attention(query, key, value, adj_matrix, mask=mask, dropout=self.dropout)

        # 4) "Concat" using a view and apply a final linear.
        node_hidden = node_hidden.transpose(1, 2).contiguous().view(n_batches, max_length, self.h * self.d_k)
        edge_hidden = edge_hidden.permute(0, 2, 3, 1, 4).contiguous().view(n_batches, max_length, max_length, self.h * self.d_k)

        return self.linears[3](node_hidden), self.linears[4](edge_hidden)


def xavier_normal_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))

    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation