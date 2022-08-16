

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

class GradInput(AttributionTechnique):
    """GradInput: Gradient times input.
    GradInput uses the gradient of a target y w.r.t its input and multiplies it
    by its input. The magnitud of the derivitate at a particular
    atom can be interpreted as a measure of how much the atom needs to be changed
    to least affect the target. Same for edges. The sign gives indication if
    this change is positive or negative. In this sense the gradient is interpreted
    as a measure of importance of each component in the input. An equation for
    this method is:
      GradInput(x) = w^T * x, where w = gradient(y w.r.t x)
    Based on "Deep Inside Convolutional Networks: Visualising Image
    Classification Models and Saliency Maps"
    (https://arxiv.org/pdf/1312.6034.pdf).
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
        # embeds = model.featurize(data)

        atom_features, atom_grads, bond_features, bond_grads = model.get_gradients(data)

        atom_weights = torch.einsum('ij,ij->i', atom_features, atom_grads) if atom_grads is not None else None
        bond_weights = torch.einsum('ij,ij->i', bond_features, bond_grads) if bond_grads is not None else None

        model.eval()
        return atom_weights, bond_weights, output
