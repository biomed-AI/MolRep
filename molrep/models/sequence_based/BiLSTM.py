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

from torch.nn.utils.rnn import pad_packed_sequence

class BiLSTM(torch.nn.Module):
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
                emb_dim     : {int} embeddings dimension
                type        : [0,1] 0-->binary_classification 1-->multiclass classification
            task_type: configs.task_type --> Classification or Regression

        Returns:
            prediction label

        Raises:
            Exception
        """
        super(BiLSTM, self).__init__()
        self.dim_features = dim_features
        self.emb_dim = model_configs['emb_dim']
        self.lstm_hid_dim = model_configs['lstm_hid_dim']
        self.hidden_dim = model_configs['hidden_dim']
        self.batch_size = model_configs['batch_size']
        self.dim_target = dim_target
        self.sen_len = 200

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
        self.linear_second = torch.nn.Linear(self.hidden_dim, self.dim_target)
        self.linear_second.bias.data.fill_(0)
        self.hidden_state = self.init_hidden()
    
    def init_hidden(self):
        return (Variable(torch.zeros(4, self.batch_size, self.lstm_hid_dim).cuda()),
                Variable(torch.zeros(4, self.batch_size, self.lstm_hid_dim)).cuda())

    def forward(self, data):
        x = data[0].long()
        embeddings = self.embeddings(x)
        self.batch_size = x.size(0)
        self.hidden_state = self.init_hidden()
        outputs, self.hidden_state = self.lstm(embeddings, self.hidden_state)  # batch * seqlenth * embed
        
        outputs = outputs[:, -1, :]
        outputs = torch.flatten(outputs, 1)
        outputs = F.tanh(self.linear_first(outputs))
        outputs = self.linear_second(outputs)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            outputs = self.sigmoid(outputs)
        if self.multiclass:
            outputs = outputs.reshape((outputs.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                outputs = self.multiclass_softmax(outputs) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return outputs