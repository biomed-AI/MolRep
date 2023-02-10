
import itertools
import functools
import logging
import warnings
from typing import Callable, List

import numpy as np
import scipy
import sklearn
import sklearn.metrics
# from graph_attribution import graphs as graph_utils


def silent_nan_np(f):
    """Decorator that silences np errors and returns nan if undefined.
    The np.nanmax and other numpy functions will log RuntimeErrors when the
    input is only nan (e.g. All-NaN axis encountered). This decorator silences
    these messages.
    Args:
      f: function to decorate.
    Returns:
      Variant of function that will be silent to invalid numpy errors, and with
      np.nan when metric is undefined.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                return f(*args, **kwargs)
            except (ValueError, sklearn.exceptions.UndefinedMetricWarning):
                return np.nan
    return wrapper


accuracy_score = sklearn.metrics.accuracy_score
nanmax = silent_nan_np(np.nanmax)
nanmean = silent_nan_np(np.nanmean)
nan_auroc_score = silent_nan_np(sklearn.metrics.roc_auc_score)
nan_precision_score = silent_nan_np(sklearn.metrics.precision_score)
nan_f1_score = silent_nan_np(sklearn.metrics.f1_score)


def nodewise_metric(f):
    """Wrapper to apply a metric computation to each nodes vector in a heatmap.
    For example, given a function `auroc` that computes AUROC over a pair
    (y_true, y_pred) for a binary classification task,
    '_heatmapwise_metric(auroc)' will compute a AUROC for each nodes heatmap in
    a list.
    Args:
      f: A function taking 1-D arrays `y_true` and `y_pred` of shape
        [num_examples] and returning some metric value.
    Returns:
      A function taking 2-D arrays `y_true` and `y_pred` of shape [num_examples,
      num_output_classes], returning an array of shape [num_output_classes].
    """

    def vectorized_f(y_true, y_pred, *args, **kwargs):
        n = len(y_true)
        values = [
            f(y_true[i].nodes, y_pred[i].nodes, *args, **kwargs) for i in range(n)
        ]
        return np.array(values)

    return vectorized_f


def _validate_attribution_inputs(y_true,
                                 y_pred):
    """Helper function to validate that attribution metric inputs are good."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f'Expected same number of graphs in y_true and y_pred, found {len(y_true)} and {len(y_pred)}'
        )
    # for att_true in y_true:
    #     node_shape = att_true.shape
    #     if len(node_shape) != 2:
    #         raise ValueError(
    #             f'Expecting 2D nodes for true attribution, found at least one with shape {node_shape}'
    #         )


def attribution_metric(f):
    """Wrapper to apply a 'attribution' style metric computation to each graph.
    For example, given a function `auroc` that computes AUROC over a pair
    (y_true, y_pred) for a binary classification task,
    '_attribution_metric(auroc)' will compute a AUROC for each graph in
    a list.
    Args:
      f: A function taking 1-D arrays `y_true` and `y_pred` of shape
        [num_examples] and returning some metric value.
    Returns:
      A function taking 2-D arrays `y_true` and `y_pred` of shape [num_examples,
      num_output_classes], returning an array of shape [num_output_classes].
    """

    def vectorized_f(y_true, y_pred, *args, **kwargs):
        _validate_attribution_inputs(y_true, y_pred)
        values = []
        for att_true, att_pred in zip(y_true, y_pred):
            values.append(f(att_true, att_pred, *args, **kwargs))
        return np.array(values)

    return vectorized_f


def kendall_tau_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kendall's tau rank correlation, used for relative orderings."""
    return scipy.stats.kendalltau(y_true, y_pred).correlation


def pearson_r_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson's r for linear correlation."""
    r, _ = scipy.stats.pearsonr(y_true, y_pred)
    return r[0] if hasattr(r, 'ndim') and r.ndim == 1 else r


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))


nodewise_f1_score = nodewise_metric(nan_f1_score)
nodewise_kendall_tau_score = nodewise_metric(kendall_tau_score)
nodewise_pearson_r_score = nodewise_metric(pearson_r_score)

attribution_auroc = attribution_metric(nan_auroc_score)
attribution_accuracy = attribution_metric(accuracy_score)
attribution_f1 = attribution_metric(nan_f1_score)
attribution_precision = attribution_metric(nan_precision_score)



def itertools_chain(a):
    return list(itertools.chain.from_iterable(a))

def attribution_accuracy_mean(y_true, y_pred):
    _validate_attribution_inputs(y_true, y_pred)
    y_true = itertools_chain(y_true)
    y_pred = itertools_chain(y_pred)
    return accuracy_score(y_true, y_pred)

def attribution_auroc_mean(y_true, y_pred):
    _validate_attribution_inputs(y_true, y_pred)
    y_true = itertools_chain(y_true)
    y_pred = itertools_chain(y_pred)
    return nan_auroc_score(y_true, y_pred)


def get_optimal_threshold(y_true,
                          y_prob,
                          grid_spacing=0.01,
                          verbose=False,
                          multi=False):
    """For probabilities, find optimal threshold according to f1 score.
    For a set of groud truth labels and predicted probabilities of these labels,
    performs a grid search over several probability thresholds. For each threshold
    f1_score is computed and the threshold that maximizes said metric is returned.
    If multiple maxmium scores are possible, we pick the median of these
    thresholds.
    Arguments:
      y_true (np.array): 1D array with true labels.
      y_prob (np.array): 1D array with predicted probabilities.
      grid_spacing (float): controls the spacing for the grid search, should be a
        positive value lower than 1.0 . Defaults to 0.01.
      verbose (bool): flag to print values.
    Returns:
      p_threshold (float): Probability threshold.
    """
    thresholds = np.arange(grid_spacing, 1.0, grid_spacing)
    scores = []
    for t in thresholds:
        if multi:
            y_preds = [np.array([1 if att>t else -1 if att<(-t) else 0 for att in att_prob]) for att_prob in y_prob]
        else:
            y_preds = [np.array([1 if att>t else 0 for att in att_prob]) for att_prob in y_prob]
        # scores.append(np.nanmean(attribution_precision(y_true, y_preds)))
        # scores.append(np.nanmean(nan_f1_score(y_true, y_preds)))
        scores.append(np.nanmean(attribution_accuracy(y_true, y_preds)))
    scores = np.array(scores)
    max_thresholds = thresholds[scores == nanmax(scores)]
    p_threshold = np.median(max_thresholds)
    if verbose:
        logging.info('Optimal p_threshold is %.2f', p_threshold)
    return p_threshold


# def get_opt_binary_attributions(atts_true,
#                                 atts_pred,
#                                 metric=nodewise_f1_score,
#                                 n_steps=20):
#     """Binarize attributions according to a threshold."""

#     thresholds = np.linspace(0, 1, num=n_steps)
#     scores = []
#     for thres in thresholds:
#         atts = [graph_utils.binarize_np_nodes(g, thres) for g in atts_pred]
#         scores.append(nanmean(metric(atts_true, atts)))
#     scores = np.array(scores)
#     max_thresholds = thresholds[scores == nanmax(scores)]
#     opt_threshold = np.median(max_thresholds)
#     atts_pred = [
#         graph_utils.binarize_np_nodes(g, opt_threshold) for g in atts_pred
#     ]
#     return atts_pred