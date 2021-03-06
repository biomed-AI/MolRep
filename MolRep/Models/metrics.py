# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica, Federico, et al. "A fair comparison of graph neural networks for graph classification." (ICLR 2020).
 -> https://github.com/diningphil/gnn-comparison
"""
import numpy as np
from math import sqrt
from typing import Dict, List, Set, Tuple, Union

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import pearsonr, spearmanr


def calc_metric(y_labels: List, y_preds: List, metric_type: str):
    # Metrics for Classifications
    if metric_type == "auc":
        return roc_auc_score(y_labels, y_preds)

    elif metric_type == "acc":
        y_preds = np.rint(y_preds)
        return accuracy_score(y_labels, y_preds)

    elif metric_type == 'prc':
        precision, recall, thresholds = precision_recall_curve(y_labels, y_preds)
        return auc(precision, recall)

    elif metric_type == 'precision':
        y_preds = np.rint(y_preds)
        return precision_score(y_labels, y_preds)

    elif metric_type == 'recall':
        y_preds = np.rint(y_preds)
        return recall_score(y_labels, y_preds)

    elif metric_type == 'f1':
        y_preds = np.rint(y_preds)
        return f1_score(y_labels, y_preds)

    elif metric_type == 'positive_pct.':
        return np.sum(y_labels) / len(y_labels)

    # Metrics for Regression
    elif metric_type == 'mse':
        return mean_squared_error(y_labels , y_preds)

    elif metric_type == "rmse":
        return sqrt(mean_squared_error(y_labels , y_preds))

    elif metric_type == "mae":
        return mean_absolute_error(y_labels, y_preds)

    elif metric_type == 'R2':
        return r2_score(y_labels, y_preds)

    elif metric_type == 'pearson':
        return pearsonr(y_labels, y_preds)[0]

    elif metric_type == 'spearman':
        return spearmanr(y_labels, y_preds)[0]
        
    elif metric_type == 'Count':
        return int(len(y_labels))

    else:
        raise Exception("There is no such metric registered")

def get_metric(y_labels: List, y_preds: List, metric_type: Union[List[str], str]):
    
    if isinstance(metric_type, List):
        res = {}
        for metric in metric_type:
            res[metric] = calc_metric(y_labels, y_preds, metric)

    elif isinstance(metric_type, str):
        res = {metric_type: calc_metric(y_labels, y_preds, metric_type)}

    else:
        raise Exception("Metric type Must be List or String")

    return res
