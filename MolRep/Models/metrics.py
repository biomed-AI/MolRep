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
import torch

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import pearsonr, spearmanr
from scipy import interp


def calc_metric(y_labels: List, y_preds: List, metric_type: str, multiclass_num_classes: int = 3):
    # Metrics for Classifications
    if metric_type == "auc":
        # Metric for Multi-Classification
        if len(y_preds[0]) > 0:
            y_preds = torch.softmax(torch.FloatTensor(y_preds), dim=1)
            y_labels = torch.nn.functional.one_hot(torch.LongTensor(y_labels), multiclass_num_classes)
            fpr, tpr, roc_auc = dict(), dict(), dict()
            for i in range(multiclass_num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_labels[:, i], y_preds[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(multiclass_num_classes)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(multiclass_num_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= multiclass_num_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            fpr["micro"], tpr["micro"], _ = roc_curve(np.array(y_labels).ravel(), np.array(y_labels).ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            return roc_auc["macro"]


        return roc_auc_score(y_labels, y_preds)

    elif metric_type == "acc":
        # Metric for Multi-Classification
        if len(y_preds[0]) > 0:
            y_preds = torch.argmax(torch.FloatTensor(y_preds), dim=1)
            return accuracy_score(y_labels, y_preds)

        y_preds = np.rint(y_preds)
        return accuracy_score(y_labels, y_preds)

    elif metric_type == 'prc':
        precision, recall, _ = precision_recall_curve(y_labels, y_preds)
        return auc(precision, recall)

    elif metric_type == 'precision':
        # Metric for Multi-Classification
        if len(y_preds[0]) > 0:
            y_preds = torch.argmax(torch.FloatTensor(y_preds), dim=1)
            return precision_score(y_labels, y_preds, average="macro")

        y_preds = np.rint(y_preds)
        return precision_score(y_labels, y_preds)

    elif metric_type == 'recall':
        # Metric for Multi-Classification
        if len(y_preds[0]) > 0:
            y_preds = torch.argmax(torch.FloatTensor(y_preds), dim=1)
            return recall_score(y_labels, y_preds, average="macro")

        y_preds = np.rint(y_preds)
        return recall_score(y_labels, y_preds)

    elif metric_type == 'f1':
        # Metric for Multi-Classification
        if len(y_preds[0]) > 0:
            y_preds = torch.argmax(torch.FloatTensor(y_preds), dim=1)
            return recall_score(y_labels, y_preds, average="macro")

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
