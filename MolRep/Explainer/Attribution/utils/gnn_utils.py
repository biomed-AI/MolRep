

import torch
import functools
import numpy as np

from rdkit import Chem
from torch_geometric import data
from MolRep.Evaluations.DatasetWrapper import Graph_data
from MolRep.Featurization.Graph_embeddings import GraphEmbeddings
from MolRep.Featurization.utils.graph_utils import *

from networkx import normalized_laplacian_matrix

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

def precompute_kron_indices(G):
    laplacians = []  # laplacian matrices (represented as 1D vectors)
    v_plus_list = []  # reduction matrices

    X = G.get_x(True, False, False)
    lap = torch.Tensor(normalized_laplacian_matrix(G).todense())  # I - D^{-1/2}AD^{-1/2}

    laplacians.append(lap)

    for _ in range(10):
        if lap.shape[0] == 1:  # Can't reduce further:
            v_plus, lap = torch.tensor([1]), torch.eye(1)
        else:
            v_plus, lap = _vertex_decimation(lap)

        laplacians.append(lap.clone())
        v_plus_list.append(v_plus.clone().long())

    return laplacians, v_plus_list

# For the Perron–Frobenius theorem, if A is > 0 for all ij then the leading eigenvector is > 0
# A Laplacian matrix is symmetric (=> diagonalizable)
# and dominant eigenvalue (true in most cases? can we enforce it?)
# => we have sufficient conditions for power method to converge
def _power_iteration(A, num_simulations=30):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = torch.rand(A.shape[1]).unsqueeze(dim=1) * 0.5 - 1

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = torch.mm(A, b_k)

        # calculate the norm
        b_k1_norm = torch.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k

def _vertex_decimation(L):

    max_eigenvec = _power_iteration(L)
    v_plus, v_minus = (max_eigenvec >= 0).squeeze(
    ), (max_eigenvec < 0).squeeze()

    # diagonal matrix, swap v_minus with v_plus not to incur in errors (does not change the matrix)
    if torch.sum(v_plus) == 0.:  # The matrix is diagonal, cannot reduce further
        if torch.sum(v_minus) == 0.:
            assert v_minus.shape[0] == L.shape[0], (v_minus.shape, L.shape)
            # assumed v_minus should have ones, but this is not necessarily the case. So added this if
            return torch.ones(v_minus.shape), L
        else:
            return v_minus, L

    L_plus_plus = L[v_plus][:, v_plus]
    L_plus_minus = L[v_plus][:, v_minus]
    L_minus_minus = L[v_minus][:, v_minus]
    L_minus_plus = L[v_minus][:, v_plus]

    L_new = L_plus_plus - \
            torch.mm(torch.mm(L_plus_minus, torch.inverse(
                L_minus_minus)), L_minus_plus)

    return v_plus, L_new


def to_data(G):
    datadict = {}
    node_features = G.get_x(True, False, False)
    datadict.update(x=node_features)

    if G.laplacians is not None:
        datadict.update(laplacians=G.laplacians)
        datadict.update(v_plus=G.v_plus)

    edge_index = G.get_edge_index()
    datadict.update(edge_index=edge_index)

    if G.number_of_edges() and G.has_edge_attrs:
        edge_attr = G.get_edge_attr()
        datadict.update(edge_attr=edge_attr)
    else:
        edge_attr = torch.Tensor([])
        datadict.update(edge_attr=edge_attr)

    target = G.get_target()
    smiles = G.get_smiles()
    datadict.update(y=target)
    datadict.update(smiles=smiles)
    datadict.update(max_num_nodes=G.max_num_nodes)

    data = Data(**datadict)
    return data


def get_gnn_data_from_smiles(smiles):

    valid_cnt, cnt = 1, 1
    valid_indices = []

    unique_node_labels = set()
    unique_edge_labels = set()

    indicator, edge_indicator = [-1], [(-1, -1)]
    graph_nodes = defaultdict(list)
    graph_edges = defaultdict(list)
    node_labels = defaultdict(list)
    edge_labels = defaultdict(list)
    node_attrs = defaultdict(list)
    edge_attrs = defaultdict(list)

    smiles_all, graph_labels = [], []
    for idx, smi in enumerate(smiles):

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        valid_indices.append(idx)
        smiles_all.append(smi)
        graph_labels.append(1)
        graph_id = valid_cnt
        valid_cnt += 1
        indicator.append(graph_id)
        
        num_nodes = len(mol.GetAtoms())

        node_dict = {}
        for atom_i, atom in enumerate(mol.GetAtoms()):
            node_dict[atom.GetIdx()] = cnt + atom_i
            graph_nodes[graph_id].append(atom_i+cnt)
            unique_node_labels.add(atom.GetAtomicNum())
            node_labels[graph_id].append(atom.GetAtomicNum())
            node_attr = atom_features(atom)
            node_attrs[graph_id].append(node_attr)
        
        for bond in mol.GetBonds():
            node_1 = node_dict[bond.GetBeginAtomIdx()]
            node_2 = node_dict[bond.GetEndAtomIdx()]
            edge_indicator.append([node_1, node_2])
            edge_indicator.append([node_2, node_1])
            graph_edges[graph_id].append([node_1, node_2])
            graph_edges[graph_id].append([node_2, node_1])
            edge_attr = [1 if i else 0 for i in bond_features(bond)]
            edge_attrs[graph_id].append(edge_attr)
            edge_attrs[graph_id].append(edge_attr)

        cnt += num_nodes
    
    graphs_data = {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "node_attrs": node_attrs,
        "edge_labels": edge_labels,
        "edge_attrs": edge_attrs,
        "smiles": smiles_all
    }
    num_node_labels = max(
        unique_node_labels) if unique_node_labels != set() else 0
    # some datasets e.g. PROTEINS have labels with value 0
    if num_node_labels != 0 and min(unique_node_labels) == 0:
        num_node_labels += 1

    num_edge_labels = max(
        unique_edge_labels) if unique_edge_labels != set() else 0
    if num_edge_labels != 0 and min(unique_edge_labels) == 0:
        num_edge_labels += 1

    targets = graphs_data.pop("graph_labels")
    smiles_list = graphs_data.pop("smiles")
    _max_num_nodes = max([len(v) for (k, v) in graphs_data['graph_nodes'].items()])

    dataset = []
    for i, (target, smi) in enumerate(zip(targets, smiles_list), 1):
        graph_data = {k: v[i] for (k, v) in graphs_data.items()}
        G = create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels, smiles=smi)
        G.max_num_nodes = _max_num_nodes
        laplacians, v_plus_list = precompute_kron_indices(G)
        G.laplacians = laplacians
        G.v_plus = v_plus_list

        data = to_data(G)
        dataset.append(data)
        G.clear()
    
    dim_features = dataset[0].x.size(1)
    dim_edge_features = dataset[0].edge_attr.size(1)

    graph_dataset = Graph_data._construct_dataset(dataset, np.arange(len(dataset)))
    graph_data = Graph_data._construct_dataloader(graph_dataset, 50, shuffle=False)

    return graph_data, valid_indices


def gnn_predict(model, test_data, scaler=None, device='cuda'):

    model.eval()
    model.to(device)

    preds = []

    for batch_index, data in enumerate(test_data):
        target_batch = data.y
        data = data.to(device)

        with torch.no_grad():
            batch_preds = model(data)

        batch_preds = batch_preds.data.cpu().numpy()
        
        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds