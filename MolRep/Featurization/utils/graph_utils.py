# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie
'''


from collections import defaultdict
import numpy as np
import networkx as nx

import torch
from torch_geometric import data
from torch_geometric.utils import dense_to_sparse

from typing import List, Tuple, Union

from rdkit import Chem

class Graph(nx.Graph):
    def __init__(self, target, smiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.smiles = smiles
        self.laplacians = None
        self.v_plus = None
        self.max_num_nodes = 200

    def get_edge_index(self):
        adj = torch.Tensor(nx.to_numpy_array(self))
        edge_index, _ = dense_to_sparse(adj)
        return edge_index

    def get_edge_attr(self):
        features = []
        for _, _, edge_attrs in self.edges(data=True):
            data = []

            if edge_attrs["label"] is not None:
                data.extend(edge_attrs["label"])

            if edge_attrs["attrs"] is not None:
                data.extend(edge_attrs["attrs"])

            features.append(data)
            features.append(data)
        return torch.Tensor(features)

    def get_x(self, use_node_attrs=False, use_node_degree=False, use_one=False, use_node_label=False):
        features = []
        
        for node, node_attrs in self.nodes(data=True):
            data = []
            if use_node_label and node_attrs["label"] is not None:  # r attention !
                data.extend(node_attrs["label"])

            if use_node_attrs and node_attrs["attrs"] is not None:
                data.extend(node_attrs["attrs"])

            if use_node_degree:
                data.extend([self.degree(node)])

            if use_one:
                data.extend([1])

            features.append(data)

        return torch.Tensor(features)

    def get_target(self):
        return np.array(self.target)

    def get_smiles(self):
        return self.smiles

    @property
    def has_edge_attrs(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["attrs"] is not None

    @property
    def has_edge_labels(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["label"] is not None

    @property
    def has_node_attrs(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["attrs"] is not None

    @property
    def has_node_labels(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["label"] is not None


def one_hot(value, num_classes):
    vec = np.zeros(num_classes)
    vec[value - 1] = 1
    return vec


def parse_tu_data(model_name, temp_dir):
    # setup paths
    indicator_path = temp_dir / f'{model_name}_graph_indicator.txt'
    edges_path = temp_dir / f'{model_name}_A.txt'
    smiles_path = temp_dir / f'{model_name}_SMILES.txt'
    graph_labels_path = temp_dir / f'{model_name}_graph_labels.txt'
    node_labels_path = temp_dir / f'{model_name}_node_labels.txt'
    edge_labels_path = temp_dir / f'{model_name}_edge_labels.txt'
    node_attrs_path = temp_dir / f'{model_name}_node_attributes.txt'
    edge_attrs_path = temp_dir / f'{model_name}_edge_attributes.txt'

    unique_node_labels = set()
    unique_edge_labels = set()

    indicator, edge_indicator = [-1], [(-1, -1)]
    graph_nodes = defaultdict(list)
    graph_edges = defaultdict(list)
    node_labels = defaultdict(list)
    edge_labels = defaultdict(list)
    node_attrs = defaultdict(list)
    edge_attrs = defaultdict(list)

    with open(indicator_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            graph_id = int(line)
            indicator.append(graph_id)
            graph_nodes[graph_id].append(i)
            
    with open(edges_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            edge = [int(e) for e in line.split(',')]
            edge_indicator.append(edge)

            # edge[0] is a node id, and it is used to retrieve
            # the corresponding graph id to which it belongs to
            # (see README.txt)
            graph_id = indicator[edge[0]]
            graph_edges[graph_id].append(edge)

    if node_labels_path.exists():
        with open(node_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                node_label = int(line)
                unique_node_labels.add(node_label)
                graph_id = indicator[i]
                node_labels[graph_id].append(node_label)

    if edge_labels_path.exists():
        with open(edge_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                edge_label = int(line)
                unique_edge_labels.add(edge_label)
                graph_id = indicator[edge_indicator[i][0]]
                edge_labels[graph_id].append(edge_label)

    if node_attrs_path.exists():
        with open(node_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                node_attr = np.array([float(n) for n in nums])
                graph_id = indicator[i]
                node_attrs[graph_id].append(node_attr)

    if edge_attrs_path.exists():
        with open(edge_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                edge_attr = np.array([float(n) for n in nums])
                graph_id = indicator[edge_indicator[i][0]]
                edge_attrs[graph_id].append(edge_attr)

    # get graph labels
    graph_labels = []
    with open(graph_labels_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            nums = line.split(",")
            targets = np.array([np.nan if n == 'None' else float(n) for n in nums])
            graph_labels.append(targets)

        # Shift by one to the left. Apparently this is necessary for multiclass tasks.
        # if min(graph_labels) == 1:
        #     graph_labels = [l - 1 for l in graph_labels]

    # get SMILES
    smiles_all = []
    with open(smiles_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            smiles_all.append(line)

    num_node_labels = max(
        unique_node_labels) if unique_node_labels != set() else 0
    # some datasets e.g. PROTEINS have labels with value 0
    if num_node_labels != 0 and min(unique_node_labels) == 0:
        num_node_labels += 1

    num_edge_labels = max(
        unique_edge_labels) if unique_edge_labels != set() else 0
    if num_edge_labels != 0 and min(unique_edge_labels) == 0:
        num_edge_labels += 1

    return {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "node_attrs": node_attrs,
        "edge_labels": edge_labels,
        "edge_attrs": edge_attrs,
        "smiles": smiles_all
    }, num_node_labels, num_edge_labels


def create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels, smiles=None):
    nodes = graph_data["graph_nodes"]
    edges = graph_data["graph_edges"]  # y list

    G = Graph(target=target, smiles=smiles)

    for i, node in enumerate(nodes):
        label, attrs = None, None

        if graph_data["node_labels"] != []:
            label = one_hot(graph_data["node_labels"][i], num_node_labels)

        if graph_data["node_attrs"] != []:
            attrs = graph_data["node_attrs"][i]

        G.add_node(node, label=label, attrs=attrs)

    for i, edge in enumerate(edges):  # y 遍历某个图的所有边
        n1, n2 = edge
        label, attrs = None, None

        if graph_data["edge_labels"] != []:
            label = one_hot(graph_data["edge_labels"][i], num_edge_labels)
        if graph_data["edge_attrs"] != []:
            attrs = graph_data["edge_attrs"][i]

        G.add_edge(n1, n2, label=label, attrs=attrs)
    
    return G


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

    def set_targets(self, target):
        self.y = target


# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.
    Args:
        value: The value for which the encoding should be one.
        choices: A list of possible values.
    
    return: 
        A one-hot encoding of the value in a list of length len(choices) + 1.
        If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    Args:
        atom: An RDKit atom.
        functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    
    return: 
        A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    Args:
        bond: A RDKit bond.
    
    return: 
        A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond