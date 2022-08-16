"""
Created on 2020.05.19

@author: Jiahua Rao, Youyang Deng, Weiming Li, Hui Yang, Jiancong Xie

Code based on: 
Zheng, Shuangjia, et al. "Identifying structure–property relationships through SMILES syntax analysis with self-attention mechanism." Journal of chemical information and modeling 59.2 (2019): 914-923.
"""


import re
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SALSTM(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, max_num_nodes=200):
        """
        Initializes parameters suggested in paper
        Args:
            dim_feature: {int} size of the vocabulary
            dim_target   : {int} number of classes
            configs:
                batch_size  : {int} batch_size used for training
                lstm_hid_dim: {int} hidden dimension for lstm
                hidden_dim  : {int} hidden dimension for the dense layer
                r           : {int} attention-hops or attention heads
                emb_dim     : {int} embeddings dimension
                type        : [0,1] 0-->binary_classification 1-->multiclass classification
            task_type: configs.task_type --> Classification or Regression

        Returns:
            prediction label

        Raises:
            Exception
        """
        super(SALSTM, self).__init__()
        self.dim_features = dim_features
        self.emb_dim = model_configs['emb_dim']
        self.lstm_hid_dim = model_configs['lstm_hid_dim']
        self.hidden_dim = model_configs['hidden_dim']
        self.batch_size = model_configs['batch_size']
        self.r = model_configs['att_hops']
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
        assert not (self.classification and self.regression and self.multiclass)

        self.embeddings = nn.Embedding(self.dim_features, self.emb_dim)
        self.lstm = torch.nn.LSTM(self.emb_dim, self.lstm_hid_dim, 2, batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(2 * self.lstm_hid_dim, self.hidden_dim)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(self.hidden_dim, self.r)
        self.linear_second.bias.data.fill_(0)
        self.linear_final = torch.nn.Linear(2 * self.lstm_hid_dim, self.dim_target)
        self.hidden_state = self.init_hidden()

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def init_hidden(self):
        return (Variable(torch.zeros(4, self.batch_size, self.lstm_hid_dim).cuda()),
                Variable(torch.zeros(4, self.batch_size, self.lstm_hid_dim)).cuda())

    def forward(self, data):
        x = data[0].long()
        embeddings = self.embeddings(x)
        self.batch_size = x.size(0)
        self.hidden_state = self.init_hidden()
        outputs, self.hidden_state = self.lstm(embeddings, self.hidden_state)  # batch * seqlenth * embed
        x = F.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r

        output = self.linear_final(avg_sentence_embeddings)
        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output