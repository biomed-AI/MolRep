

import abc
from copy import deepcopy
import torch
import functools

import numpy as np
from typing import Text, Tuple

from torch_geometric import data
from MolRep.Evaluations.DatasetWrapper import Graph_data, MPNN_data


class Data(data.Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 edge_attr=None,
                 y=None,
                 v_outs=None,
                 e_outs=None,
                 g_outs=None,
                 o_outs=None,
                 laplacians=None,
                 v_plus=None,
                 smiles=None,
                 max_num_nodes=200,
                 **kwargs):

        additional_fields = {
            'v_outs': v_outs,
            'e_outs': e_outs,
            'g_outs': g_outs,
            'o_outs': o_outs,
            'laplacians': laplacians,
            'v_plus': v_plus,
            'max_num_nodes': max_num_nodes,
            'smiles': smiles
        }
        super().__init__(x, edge_index, edge_attr, y, **additional_fields)


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


class IntegratedGradients(AttributionTechnique):
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


    def __init__(self,
                 num_steps: int = 200,
                 name = None):
        """Constructor for IntegratedGradients.
        """
        self.name = name or self.__class__.__name__
        self.num_steps = num_steps
        self.sample_size = num_steps


    def attribute(self, data,
                        model,
                        model_name,
                        scaler=None):
        
        model.train()

        output = model(data)
        if not isinstance(output, tuple):
            output = (output,)

        if model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'GAT']:
            node_null = np.zeros((1, data.x.size()[1]))
            edge_null = np.zeros((1, data.edge_attr.size()[1])) if data.edge_attr is not None else None
            self.reference_fn = self.make_reference_fn(node_null, edge_null)

            n = self.num_steps
            ref = self.reference_fn(data)
            n_nodes = data.x.shape[0]
            n_edges = data.edge_attr.shape[0] if data.edge_attr is not None else 0
            # print(node_null.shape, data.x.size(), ref.x.size())
            interp_data, node_steps, edge_steps = self.interpolate_graphs(ref, data, n, model_name)

            _, atom_grads, _, bond_grads = [model.get_gradients(data) for data in interp_data][0]
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

        elif ['MPNN', 'DMPNN', 'CMPNN']:
            atom_features, atom_grads, bond_features, bond_grads = model.get_gradients(data)
            
            atom_weights = torch.einsum('ij,ij->i', atom_features, atom_grads) if atom_grads is not None else None
            bond_weights = torch.einsum('ij,ij->i', bond_features, bond_grads) if bond_grads is not None else None

        model.eval()
        return atom_weights, bond_weights, output


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
                               , model_name: str = ''):

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
        
        if model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'GAT']:
            interp_dataset = Graph_data._construct_dataset(interp_graphs, np.arange(len(interp_graphs)))
            interp_data = Graph_data._construct_dataloader(interp_dataset, len(interp_graphs), shuffle=False)
        elif model_name in ['MPNN', 'DMPNN', 'CMPNN']:
            interp_dataset = MPNN_data._construct_dataset(interp_graphs, np.arange(len(interp_graphs)))
            interp_data = MPNN_data._construct_dataloader(interp_dataset, len(interp_graphs), shuffle=False)
        else:
            raise print("Explainer Model Must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'MPNN', 'DMPNN', 'CMPNN']")
        return interp_data, node_steps, edge_steps

