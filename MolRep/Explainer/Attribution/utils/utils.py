
import collections
import functools
from typing import Callable, List, MutableMapping, Optional, Text

import torch
import numpy as np

from MolRep.Experiments.Graph_Data.Graph_data import Batch

def make_constant_like(data, node_vec: np.ndarray,
                       edge_vec: np.ndarray):
    """Make a similar graph but with constant nodes and edges."""
    using_tensors = isinstance(data.x, torch.Tensor)
    nodes = np.tile(node_vec, (data.x.shape[0], 1)) if data.x is not None else None
    edges = np.tile(edge_vec, (data.edge_attr.shape[0], 1)) if data.edge_attr is not None else None
    if using_tensors:
        nodes = torch.FloatTensor(nodes)
        edges = torch.FloatTensor(edges)
    ref_data = data
    ref_data.x = nodes
    ref_data.edge_attrs = edges
    return ref_data

def make_reference_fn(node_vec: np.ndarray, edge_vec: np.ndarray):
    """Make reference function."""
    ref_fn = functools.partial(
        make_constant_like, node_vec=node_vec, edge_vec=edge_vec)
    return ref_fn


def _interp_array(start: torch.Tensor, end: torch.Tensor,
                  num_steps: int) -> torch.Tensor:
    """Linearly interpolate 2D tensors, returns 3D tensors.
    Args:
      start: 2D tensor for start point of interpolation of shape [x,y].
      end: 2D tensor as end point of interpolation of shape [x,y] (same as start).
      num_steps: number of steps to interpolate.
    Returns:
      New tensor of shape [num_steps, x, y]
    """
    alpha = torch.linspace(0., 1., num_steps)
    beta = 1 - alpha
    return torch.einsum('a,bc->abc', alpha, end) + torch.einsum('a,bc->abc', beta,
                                                          start)


def interpolate_graphs_tuple(
        start, end,
        num_steps: int,
        device = 'cuda'):
    """Interpolate two graphs of same shape."""
    nodes_interp = _interp_array(start.x, end.x.cpu(), num_steps)
    edges_interp = _interp_array(start.edge_attr.cpu(), end.edge_attr.cpu(), num_steps)
    node_steps = torch.tile(nodes_interp[1] - nodes_interp[0], (num_steps, 1))
    edge_steps = torch.tile(edges_interp[1] - edges_interp[0], (num_steps, 1))
    x_list, edge_attrs_list = [], []
    for nodes, edges in zip(nodes_interp, edges_interp):
        x_list.append(nodes)
        edge_attrs_list.append(edges)

    interp_data = end
    interp_data.x = torch.mean(torch.stack(x_list,dim=0),dim=0)
    interp_data.edge_attr = torch.mean(torch.stack(edge_attrs_list,dim=0),dim=0)
    return interp_data.to(device), node_steps, edge_steps