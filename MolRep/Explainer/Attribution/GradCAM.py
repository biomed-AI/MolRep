



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



class GradCAM(AttributionTechnique):
    """GradCAM: intermediate activations and gradients as input importance.
    GradCAM is the gradient version of CAM using ideas from Gradient times Input,
    removing the necessity of a GAP layer.
    For each convolution layer, in the case of graphs a GNN block, the
    activations can be retrieved and interpreted as a transformed version of the
    input. In a GNN intermediate activations are graphs with updated information.
    The gradient of a target y w.r.t these activations can be seen as measure of
    importance. The equation for gradCAM are:
      GradCAM(x) = reduce_i mean(w_i^T G_i(x), axis=-1)
    G_i(x) is the intermediate layer activations.
    reduce_i is an reduction operation over intermediate layers (e.g. mean, sum).
    Based on "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization" (https://arxiv.org/abs/1610.02391).
    """

    def __init__(self,
                 last_layer_only: bool = True,
                 reduce_fn=torch.mean,
                 name: Optional[Text] = None):
        """GradCAM constructor.
        Args:
          last_layer_only: If to use only the last layer activations, if not will
            use all last activations.
          reduce_fn: Reduction operation for layers, should have the same call
            signature as torch.mean (e.g. tf.reduce_sum).
          name: identifying label for method.
        """
        self.name = name or self.__class__.__name__
        self.last_layer_only = last_layer_only
        self.sample_size = 1
        self.reduce_fn = reduce_fn
        try:
            reduce_fn(torch.Tensor([[0], [1]]), dim=0)
        except BaseException:
            raise ValueError(
                'reduce_fn should have a signature like tf.reduce_mean!')

    def attribute(self, data, model, model_name, scaler=None):
        """Gets attribtutions."""
        model.train()

        output = model(data)
        if not isinstance(output, tuple):
            output = (output,)

        acts, grads = model.get_intermediate_activations_gradients(data)
        node_w, edge_w = [], None
        layer_indices = [-1] if self.last_layer_only else list(range(len(acts)))
        for index in layer_indices:
            node_act = acts[index]
            node_grad = grads[index]
            node_w.append(torch.einsum('ij,ij->i', node_act, node_grad).reshape(-1,1))

        node_weights = self.reduce_fn(torch.cat(node_w, dim=1), dim=1) if node_w is not None else None
        edge_weights = self.reduce_fn(torch.cat(edge_w, dim=1), dim=1) if edge_w is not None else None
        
        model.eval()
        return node_weights, edge_weights, output