# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica, Federico, et al. "A fair comparison of graph neural networks for graph classification." (ICLR 2020).
 -> https://github.com/diningphil/gnn-comparison
"""
from math import sqrt

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_metric(y_labels, y_preds, metric_type: str):
    if metric_type == "auc":
        return roc_auc_score(y_labels, y_preds)
    elif metric_type == "acc":
        return accuracy_score(y_labels, y_preds)
    elif metric_type == 'prc':
        precision, recall, thresholds = precision_recall_curve(y_labels, y_preds)
        return auc(precision, recall)
    elif metric_type == "mse":
        return mean_squared_error(y_labels , y_preds)
    elif metric_type == "rmse":
        return sqrt(mean_squared_error(y_labels , y_preds))
    elif metric_type == "mae":
        return mean_absolute_error(y_labels, y_preds)
    else:
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_labels, y_preds, average="micro")
        if metric_type == "all":
            acc = accuracy_score(y_labels, y_preds)
            auc = roc_auc_score(y_labels, y_preds)
            return precision, recall, fscore, acc, auc
        elif metric_type == "fscore":
            return fscore
        elif metric_type == "precision":
            return precision
        elif metric_type == "recall":
            return recall
    raise Exception("There is no such metric registered")
