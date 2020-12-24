# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Liu et al " N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules." -> https://github.com/chao1224/n_gram_graph
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd


def N_Gram_Graph_construct_loader(features_path, train_idxs=None, valid_idxs=None, test_idxs=None, task_type='Classification', features_scaling=True):
    x_all, y_all = pickle.load(open(features_path, "rb"))

    train_x, train_y = (np.array(x_all)[train_idxs], np.array(y_all)[train_idxs]) if train_idxs is not None else (None, None)
    if task_type == 'Regression' and train_idxs is not None:
        train_targets = train_y
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_y = scaled_targets
    else:
        scaler = None

    train_loader = (train_x, train_y) if train_idxs is not None else None
    valid_loader = (np.array(x_all)[valid_idxs], np.array(y_all)[valid_idxs]) if valid_idxs is not None else None
    test_loader = (np.array(x_all)[test_idxs], np.array(y_all)[test_idxs]) if test_idxs is not None else None

    return train_loader, valid_loader, test_loader, None, scaler