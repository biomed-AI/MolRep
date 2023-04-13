

from copy import deepcopy
import torch
import functools
from pathlib import Path

import numpy as np
from typing import Tuple

from torch_geometric.utils import degree
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

from molrep.explainer.base_explainer import BaseExplainer
from molrep.common.registry import registry

@registry.register_explainer("ig")
class IntegratedGradients(BaseExplainer):
    r"""IG: path intergral between a graph and a counterfactual.
    Because IntegratedGradients is based on path integrals, it has nice
    properties associated to integrals, namely IG(x+y) = IG(x)+ IG(y) and
    if y is a target to predict, then y(x) - y(ref) = sum(IG(x,ref)). This last
    property is useful for sanity checks and also indicates that the difference
    in predictions can be retrieved from the attribution.
      IG(x,ref) = \integral_{1}^{0} grad(y w.r.t. interp(ref,x,t))*stepsize   t
      where stepsize = (x-ref)/n_steps.
    From the paper "Axiomatic Attribution for Deep Networks"
    (https://arxiv.org/abs/1703.01365) and "Using attribution to decode binding
    mechanism in neural network models for chemistry"
    (https://www.pnas.org/content/116/24/11624).
    """

    model_processer_mapping = {
        "mpnn": "mpnn", "dmpnn": "mpnn", "cmpnn": "mpnn",
        "graphsage": "graph", "graphnet": "graph", "gin": "graph",
        "bilstm": "sequence", "salstm": "sequence", "transformer": "sequence",
    }

    def __init__(self,
                 num_steps: int = 200,
                 name = None):
        """Constructor for IntegratedGradients.
        """
        self.name = name or self.__class__.__name__
        self.num_steps = num_steps
        self.sample_size = num_steps


    def attribute(self, data, model, **kwargs):
        model.train()
        config = kwargs['config']
        if isinstance(data, dict):
            data = data["pygdata"]

        n_nodes = data.x.shape[0]
        n_edges = data.edge_attr.shape[0] if data.edge_attr is not None else 0

        node_null = np.zeros((1, data.x.size()[1]))
        edge_null = np.zeros((1, data.edge_attr.size()[1])) if data.edge_attr is not None else None
        self.reference_fn = self.make_reference_fn(node_null, edge_null)

        n = self.num_steps

        sizes = degree(data.batch, dtype=torch.long).tolist()
        node_feat_list = data.x.split(sizes, dim=0)
        assert len(node_feat_list) == 1, "Batch size of IntegratedGradients should be set to 1."    #TODO

        ref = self.reference_fn(data)
        # print(node_null.shape, data.x.size(), ref.x.size())
        interp_data, node_steps, edge_steps = self.interpolate_graphs(ref, data, n, config)

        _, atom_grads, _, bond_grads = model.get_gradients(interp_data)
        # Node shapes: [n_nodes * n, nodes.shape[-1]] -> [n_nodes*n].
        atom_grads = torch.tensor(atom_grads, dtype=torch.float)
        node_steps = torch.tensor(node_steps, dtype=torch.float, device=atom_grads.device)
        node_values = torch.einsum('ij,ij->i', atom_grads, node_steps)
        # Node shapes: [n_nodes * n] -> [n_nodes, n].
        node_values = torch.reshape(node_values, (n, n_nodes)).t()
        # Node shapes: [n_nodes, n] -> [n_nodes].
        atom_weights = torch.sum(node_values, axis=1)

        if bond_grads is not None:
            bond_grads = torch.tensor(bond_grads, dtype=torch.float)
            edge_steps = torch.tensor(edge_steps, dtype=torch.float, device=bond_grads.device)
            edge_values = torch.einsum('ij,ij->i', bond_grads, edge_steps)
            edge_values = torch.reshape(edge_values, (n, n_edges)).t()
            bond_weights = torch.sum(edge_values, axis=1)
        else:
            bond_weights = None

        model.eval()
        return atom_weights, bond_weights

    def make_reference_fn(self, node_vec, edge_vec):
        """Make reference function."""
        ref_fn = functools.partial(
            self.make_constant_like, node_vec=node_vec, edge_vec=edge_vec)
        return ref_fn

    def make_constant_like(self, data, node_vec = None, edge_vec = None):
        """Make a similar graph but with constant nodes and edges."""
        using_tensors = isinstance(data.x, torch.Tensor)
        nodes = np.tile(node_vec, (data.x.shape[0], 1))
        edges = np.tile(edge_vec, (data.edge_attr.shape[0], 1)) if edge_vec is not None else None
        if using_tensors:
            nodes = torch.tensor(nodes, dtype=data.x.dtype, device=data.x.device)
            edges = torch.tensor(edges, dtype=data.edge_attr.dtype, device=data.edge_attr.device)
        ref_data = deepcopy(data)
        ref_data.x = nodes
        ref_data.edge_attr = edges
        return ref_data

    def _interp_array(self, start: np.ndarray, end: np.ndarray,
                      num_steps: int):
        """Linearly interpolate 2D tensors, returns 3D tensors.
        Args:
          start: 2D tensor for start point of interpolation of shape [x,y].
          end: 2D tensor as end point of interpolation of shape [x,y] (same as start).
          num_steps: number of steps to interpolate.
        Returns:
          New tensor of shape [num_steps, x, y]
        """
        alpha = np.linspace(0., 1., num_steps)
        beta = 1 - alpha
        return np.einsum('a,bc->abc', alpha, end) + np.einsum('a,bc->abc', beta,
                                                              start)

    def interpolate_graphs(self, start: Tuple = None
                               , end: Tuple = None
                               , num_steps: int = 50
                               , config = None):
        model_name = config.model_cfg.name
        dataset_cls = registry.get_dataset_class((self.model_processer_mapping[model_name]))

        nodes_interp = self._interp_array(start.x.cpu().detach().numpy(), end.x.cpu().detach().numpy(), num_steps)
        edges_interp = self._interp_array(start.edge_attr.cpu().detach().numpy(), end.edge_attr.cpu().detach().numpy(), num_steps)
        node_steps = np.tile(nodes_interp[1] - nodes_interp[0], (num_steps, 1))
        edge_steps = np.tile(edges_interp[1] - edges_interp[0], (num_steps, 1))
        interp_graphs = []
        for nodes, edges in zip(nodes_interp, edges_interp):
            datadict = end.to_dict()
            datadict['x'] = torch.tensor(nodes, dtype=datadict['x'].dtype, device=datadict['x'].device)
            datadict['edge_attr'] = torch.tensor(edges, dtype=datadict['edge_attr'].dtype, device=datadict['edge_attr'].device)
            interp_graphs.append(Data(**datadict))

        interp_dataset = dataset_cls(interp_graphs)
        interp_data = DataLoader(
                        dataset=interp_dataset,
                        batch_size=len(interp_graphs),
                        collate_fn=lambda data_list: self.collate_fn(data_list),
                        shuffle=False
        )

        return interp_data, node_steps, edge_steps

    def collate_fn(data):
        batch_data = Batch.from_data_list(data)
        return {
            "pygdata": batch_data,
            "targets": batch_data.y,
        }
