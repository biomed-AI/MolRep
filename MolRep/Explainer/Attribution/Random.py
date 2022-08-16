
import abc
import torch
import numpy as np
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


class RandomBaseline(AttributionTechnique):
    """Random baseline: random node and edge attributions from uniform(0,1)."""

    def __init__(self, name: Optional[Text] = None):
        self.name = name or self.__class__.__name__
        self.sample_size = 1

    def attribute(self,data, model, model_name, scaler=None):
        """Gets attribtutions."""
        model.train()
        
        output = model(data)
        if not isinstance(output, tuple):
            output = (output,)

        # Prediction
        if model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'PyGCMPNN', 'GAT', 'MolecularFingerprint']:

            atom_weights = torch.Tensor(np.random.uniform(size=(data.x.shape[0]))) if data.x is not None else None
            bond_weights = torch.Tensor(np.random.uniform(size=(data.edge_attr.shape[0]))) if data.edge_attr is not None else None

        elif model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            atom_features, atom_grads, bond_features, bond_grads = model.get_gradients(data)

            atom_weights = torch.Tensor(np.random.uniform(size=(atom_grads.shape[0]))) if atom_grads is not None else None
            bond_weights = torch.Tensor(np.random.uniform(size=(bond_grads.shape[0]))) if bond_grads is not None else None

        model.eval()
        return atom_weights, bond_weights, output