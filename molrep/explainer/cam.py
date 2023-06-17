
import torch
from typing import Optional, Text

from molrep.explainer.base_explainer import BaseExplainer
from molrep.common.registry import registry


@registry.register_explainer("cam")
class CAM(BaseExplainer):
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

    EXPLAINER_CONFIG_DICT = {
        "default": "configs/explainer/cam_default.yaml",
    }

    def __init__(self, name: Optional[Text] = None):
        self.name = name or self.__class__.__name__
        self.sample_size = 1

    def explain(self, data, model, **kwargs):
        if isinstance(data, dict):
            data = data["pygdata"]

        """Gets attribtutions."""
        model.train()

        node_act, edge_act = model.get_gap_activations(data)
        weights = model.get_prediction_weights()

        node_weights = torch.einsum('ij,j', node_act, weights) if node_act is not None else None
        edge_weights = torch.einsum('ij,j', edge_act, weights) if edge_act is not None else None
        
        if node_weights is not None:
            node_weights = model.unbatch(node_weights, data, is_atom=True)
        
        if edge_weights is not None:
            edge_weights = model.unbatch(edge_weights, data, is_atom=False)

        model.eval()
        return node_weights, edge_weights