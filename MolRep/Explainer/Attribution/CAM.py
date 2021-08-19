

import abc
import torch
from typing import Any, Callable, List, MutableMapping, Optional, Text, Tuple

class AttributionTechnique(abc.ABC):
    """Abstract class for an attribution technique."""

    name: Text
    sample_size: int  # Number of graphs to hold in memory per input.
    

    @abc.abstractmethod
    def attribute(self, data, model, model_name):
        """Compute GraphTuple with node and edges importances.
        Assumes that x (GraphTuple) has node and edge information as 2D arrays
        and the returned attribution will be a list of GraphsTuple, for each
        graph inside of x, with the same shape but with 1D node and edge arrays.
        Args:
          x: Input to get attributions for.
          model: model that gives gradients, predictions, activations, etc.
          task_index: index for task to focus attribution.
          batch_index: index for example to focus attribution.
        """


class CAM(AttributionTechnique):
    """CAM: Decompose output as a linear sum of nodes and edges.
    CAM (Class Activation Maps) assumes the model has a global average pooling
    layer (GAP-layer) right before prediction. This means the prediction can be
    written as weighted sum of the pooled elements plus an final activation.
    In the case of graphs, a GAP layer should take nodes and edges activations
    and will sum them to create a graph embedding layer. The CAM model follows
    the equation:
      CAM(x) = (node_activations + edge_activations)*w
    Based on "Learning Deep Features for Discriminative Localization"
    (https://arxiv.org/abs/1512.04150).
    """

    def __init__(self, name: Optional[Text] = None):
        self.name = name or self.__class__.__name__
        self.sample_size = 1

    def attribute(self, data, model, model_name, scaler=None):
        """Gets attribtutions."""
        model.train()

        output = model(data)
        if not isinstance(output, tuple):
            output = (output,)

        node_act, edge_act = model.get_gap_activations(data)
        weights = model.get_prediction_weights()

        # print('node_act:', node_act.shape)
        # print('weights:', weights.shape)
        node_weights = torch.einsum('ij,j', node_act, weights) if node_act is not None else None
        edge_weights = torch.einsum('ij,j', edge_act, weights) if edge_act is not None else None
        
        model.eval()
        return node_weights, edge_weights, output