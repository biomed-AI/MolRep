import os
import copy
import pickle
from multiprocessing import Pool
from itertools import combinations
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import inv, eigs
import networkx as nx
from sknetwork.embedding import Spectral
from sknetwork.utils import membership_matrix
from sknetwork.hierarchy import Ward, cut_straight
from sknetwork.clustering import Louvain, KMeans, PropagationClustering
from geomloss import SamplesLoss
import MolRep.Interactions.link_models.CFLP.pysbm as pysbm


def load_t_files(model_configs, dataset_configs, T_file, adj_train):
    # raw node embeddings for nearest neighbor finding: numpy.ndarray
    node_embs_raw = pickle.load(open(f"{dataset_configs['path']}/{dataset_configs['name']}_embs-raw{model_configs['embraw']}.pkl", 'rb'))
    # print('cf distance threshold: ', np.percentile(cdist(node_embs_raw, node_embs_raw, 'euclidean'), args.gamma))
    if os.path.exists(T_file):
        T_f, T_cf, adj_cf, edges_cf_t0, edges_cf_t1 = pickle.load(open(T_file, 'rb'))
    else:
        T_f = get_t(adj_train, model_configs['t'], model_configs['k'], model_configs['selfloopT'])
        T_cf, adj_cf, edges_cf_t0, edges_cf_t1 = get_CF(adj_train, node_embs_raw, T_f, model_configs['dist'], model_configs['gamma'], 8)
        T_cf = sp.csr_matrix(T_cf)
        adj_cf = sp.csr_matrix(adj_cf)
        pickle.dump((T_f, T_cf, adj_cf, edges_cf_t0, edges_cf_t1), open(T_file, 'wb'))
    return T_f, edges_cf_t1, edges_cf_t0, T_cf, adj_cf


def get_t(adj_mat, method, k, selfloop=False):
    adj = copy.deepcopy(adj_mat)
    if not selfloop:
        adj.setdiag(0)
        adj.eliminate_zeros()
    if method == 'anchor_nodes':
        T = anchor_nodes(adj, k)
    elif method == 'common_neighbors':
        T = common_neighbors(adj, k)
    elif method == 'louvain':
        T = louvain(adj)
    elif method == 'spectral_clustering':
        T = spectral_clustering(adj, k)
    elif method == 'propagation':
        T = propagation(adj)
    elif method == 'kcore':
        T = kcore(adj)
    elif method == 'katz':
        T = katz(adj, k)
    elif method == 'hierarchy':
        T = ward_hierarchy(adj, k)
    elif method == 'jaccard':
        T = jaccard_index(adj, k)
    elif method == 'sbm':
        T = SBM(adj, k)
    return T

def SBM(adj, k):
    nx_g = nx.from_scipy_sparse_matrix(adj)
    standard_partition = pysbm.NxPartition(graph=nx_g, number_of_blocks=k)
    rep = standard_partition.get_representation()
    labels = np.asarray([v for k, v in sorted(rep.items(), key=lambda item: item[0])])
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def ward_hierarchy(adj, k):
    ward = Ward()
    dendrogram = ward.fit_transform(adj)
    labels = cut_straight(dendrogram, k)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def jaccard_index(adj, k):
    adj = adj.astype(int)
    intrsct = adj.dot(adj.T)
    row_sums = intrsct.diagonal()
    unions = row_sums[:,None] + row_sums - intrsct
    sim_matrix = intrsct / unions
    thre = np.percentile(sim_matrix, (100-10*k))
    thre = max(thre, np.percentile(sim_matrix, 0.5))
    thre = min(thre, np.percentile(sim_matrix, 0.8))
    T = np.asarray((sim_matrix >= thre).astype(int))
    T = T - np.diag(T.diagonal())
    return sp.csr_matrix(T)

def katz(adj, k):
    max_eigvalue = eigs(adj.astype(float), k=1)[0][0]
    beta = min(1/max_eigvalue/2, 0.003)
    sim_matrix = inv(sp.identity(adj.shape[0]) - beta *  adj) - sp.identity(adj.shape[0])
    sim_matrix = sim_matrix.toarray()
    size = sim_matrix.shape[0]
    thre = 2 * k * sim_matrix.sum() / (size*size-1)
    T = np.asarray((sim_matrix > thre).astype(int))
    T = T - np.diag(T.diagonal())
    return sp.csr_matrix(T)

def anchor_nodes(adj, k):
    row_sum = np.asarray(adj.sum(axis = 1)).reshape(-1)
    dist = dijkstra(csgraph=adj, indices=np.argmax(row_sum), directed=False, limit=k+1, return_predecessors=False)
    res = dist < (k+1)
    T = np.zeros(adj.shape)
    T[res] += 1
    T[:,res] += 1
    T = (T > 1).astype(int)
    return sp.csr_matrix(T)

def common_neighbors(adj, k):
    mul_hop_adj = adj
    for i in range(2):
        mul_hop_adj += adj ** (i+2)
    mul_hop_adj = (mul_hop_adj>0).astype(int)
    T = (mul_hop_adj @ mul_hop_adj.T) >= k
    T = T.astype(int)
    return T

def louvain(adj):
    louvain = Louvain()
    labels = louvain.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def propagation(adj):
    propagation = PropagationClustering()
    labels = propagation.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def spectral_clustering(adj, k):
    kmeans = KMeans(n_clusters = k, embedding_method=Spectral(256))
    labels = kmeans.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def kcore(adj):
    G = nx.from_scipy_sparse_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    labels = np.array(list(nx.algorithms.core.core_number(G).values()))-1
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def sample_nodepairs(num_np, edges_f_t1, edges_f_t0, edges_cf_t1, edges_cf_t0):
    # TODO: add sampling with separated treatments
    nodepairs_f = np.concatenate((edges_f_t1, edges_f_t0), axis=0)
    f_idx = np.random.choice(len(nodepairs_f), min(num_np,len(nodepairs_f)), replace=False)
    np_f = nodepairs_f[f_idx]
    nodepairs_cf = np.concatenate((edges_cf_t1, edges_cf_t0), axis=0)
    cf_idx = np.random.choice(len(nodepairs_cf), min(num_np,len(nodepairs_f)), replace=False)
    np_cf = nodepairs_cf[cf_idx]
    return np_f, np_cf

def calc_disc(disc_func, z, nodepairs_f, nodepairs_cf):
    X_f = torch.cat((z[nodepairs_f.T[0]], z[nodepairs_f.T[1]]), axis=1)
    X_cf = torch.cat((z[nodepairs_cf.T[0]], z[nodepairs_cf.T[1]]), axis=1)
    if disc_func == 'lin':
        mean_f = X_f.mean(0)
        mean_cf = X_cf.mean(0)
        loss_disc = torch.sqrt(F.mse_loss(mean_f, mean_cf) + 1e-6)
    elif disc_func == 'kl':
        # TODO: kl divergence
        pass
    elif disc_func == 'w':
        # Wasserstein distance
        dist = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        loss_disc = dist(X_cf, X_f)
    else:
        raise Exception('unsupported distance function for discrepancy loss')
    return loss_disc

def get_CF(adj, node_embs, T_f, dist='euclidean', thresh=50, n_workers=20):
    if dist == 'cosine':
        # cosine similarity (flipped to use as a distance measure)
        embs = normalize(node_embs, norm='l1', axis=1)
        simi_mat = embs @ embs.T
        simi_mat = 1 - simi_mat
    elif dist == 'euclidean':
        # Euclidean distance
        simi_mat = cdist(node_embs, node_embs, 'euclidean')
    thresh = np.percentile(simi_mat, thresh)
    # give selfloop largest distance
    np.fill_diagonal(simi_mat, np.max(simi_mat)+1)
    # nearest neighbor nodes index for each node
    node_nns = np.argsort(simi_mat, axis=1)
    # find nearest CF node-pair for each node-pair
    node_pairs = list(combinations(range(adj.shape[0]), 2))
    print('This step may be slow, please adjust args.n_workers according to your machine')
    pool = Pool(n_workers)
    batches = np.array_split(node_pairs, n_workers)
    results = pool.map(get_CF_single, [(adj, simi_mat, node_nns, T_f, thresh, np_batch, True) for np_batch in batches])
    results = list(zip(*results))
    T_cf = np.add.reduce(results[0])
    adj_cf = np.add.reduce(results[1])
    edges_cf_t0 = np.concatenate(results[2])
    edges_cf_t1 = np.concatenate(results[3])
    return T_cf, adj_cf, edges_cf_t0, edges_cf_t1,

def get_CF_single(params):
    """ single process for getting CF edges """
    adj, simi_mat, node_nns, T_f, thresh, node_pairs, verbose = params

    T_cf = np.zeros(adj.shape)
    adj_cf = np.zeros(adj.shape)
    edges_cf_t0 = []
    edges_cf_t1 = []
    c = 0
    for a, b in node_pairs:
        # for each node pair (a,b), find the nearest node pair (c,d)
        nns_a = node_nns[a]
        nns_b = node_nns[b]
        i, j = 0, 0
        while i < len(nns_a)-1 and j < len(nns_b)-1:
            if simi_mat[a, nns_a[i]] + simi_mat[b, nns_b[j]] > 2 * thresh:
                T_cf[a, b] = T_f[a, b]
                adj_cf[a, b] = adj[a, b]
                break
            if T_f[nns_a[i], nns_b[j]] != T_f[a, b]:
                T_cf[a, b] = 1 - T_f[a, b] # T_f[nns_a[i], nns_b[j]] when treatment not binary
                adj_cf[a, b] = adj[nns_a[i], nns_b[j]]
                if T_cf[a, b] == 0:
                    edges_cf_t0.append([nns_a[i], nns_b[j]])
                else:
                    edges_cf_t1.append([nns_a[i], nns_b[j]])
                break
            if simi_mat[a, nns_a[i+1]] < simi_mat[b, nns_b[j+1]]:
                i += 1
            else:
                j += 1
        c += 1
        if verbose and c % 20000 == 0:
            print(f'{c} / {len(node_pairs)} done')
    edges_cf_t0 = np.asarray(edges_cf_t0)
    edges_cf_t1 = np.asarray(edges_cf_t1)
    return T_cf, adj_cf, edges_cf_t0, edges_cf_t1

import os
import copy
import math
import pickle
import logging
import numpy as np
import networkx as nx
import scipy.sparse as sp
from copy import deepcopy
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
# from dgl.data.citation_graph import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from ogb.linkproppred import Evaluator#, PygLinkPropPredDataset
# from ogb.linkproppred import DglLinkPropPredDataset


def eval_ep_batched(logits, labels, n_pos, evaluator=None):
    # roc-auc and ap
    roc_auc = roc_auc_score(labels, logits)
    ap_score = average_precision_score(labels, logits)
    results = {'auc': roc_auc,
               'ap': ap_score}
    # hits@K
    if evaluator is None:
        evaluator = Evaluator(name='ogbl-ddi')
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': logits[:n_pos],
            'y_pred_neg': logits[n_pos:],
        })[f'hits@{K}']
        results[f'hits@{K}'] = hits
    return results

def eval_ep(A_pred, edges, edges_false, evaluator=None):
    preds = A_pred[edges.T]
    preds_neg = A_pred[edges_false.T]
    logits = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    # roc-auc and ap
    roc_auc = roc_auc_score(labels, logits)
    ap_score = average_precision_score(labels, logits)
    results = {'auc': roc_auc,
               'ap': ap_score}
    # hits@K
    if evaluator is None:
        evaluator = Evaluator(name='ogbl-ddi')
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': preds,
            'y_pred_neg': preds_neg,
        })[f'hits@{K}']
        results[f'hits@{K}'] = hits
    return results

def normalize_sp(adj_matrix):
    # normalize adj by D^{-1/2}AD^{-1/2} for scipy sparse matrix input
    degrees = np.array(adj_matrix.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
    degree_mat_inv_sqrt = np.nan_to_num(degree_mat_inv_sqrt)
    adj_norm = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
    return adj_norm

def mask_test_edges(adj_orig, val_frac, test_frac, filename, logger):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    if os.path.exists(filename):
        adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false = pickle.load(open(filename, 'rb'))
        logger.info(f'loaded cached val and test edges with fracs of {val_frac} and {test_frac}')
        return adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false

    # Remove diagonal elements
    adj = deepcopy(adj_orig)
    # set diag as all zero
    adj.setdiag(0)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj, 1)
    # adj_tuple = sparse_to_tuple(adj_triu)
    # edges = adj_tuple[0]
    edges = sparse_to_tuple(adj_triu)[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * val_frac))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    noedge_mask = np.ones(adj.shape) - adj_orig
    noedges = np.asarray(sp.triu(noedge_mask, 1).nonzero()).T
    all_edge_idx = list(range(noedges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges_false = noedges[test_edge_idx]
    val_edges_false = noedges[val_edge_idx]
    # following lines for getting the no-edges are substituted with above lines
    """
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    test_edges_false = np.asarray(test_edges_false).astype("int32")
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    val_edges_false = np.asarray(val_edges_false).astype("int32")
    """
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    adj_train.setdiag(1)

    # get training node pairs (edges and no-edges)
    train_mask = np.ones(adj_train.shape)
    for edges_tmp in [val_edges, val_edges_false, test_edges, test_edges_false]:
        for e in edges_tmp:
            assert e[0] < e[1]
        train_mask[edges_tmp.T[0], edges_tmp.T[1]] = 0
        train_mask[edges_tmp.T[1], edges_tmp.T[0]] = 0
    train_pairs = np.asarray(sp.triu(train_mask, 1).nonzero()).T

    # cache files for future use
    pickle.dump((adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false), open(filename, 'wb'))
    logger.info(f'masked and cached val and test edges with fracs of {val_frac} and {test_frac}')

    # NOTE: all these edge lists only contain single direction of edge!
    return adj_train, train_pairs, val_edges, val_edges_false, test_edges, test_edges_false

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, lr_scheduler, *op):
        self.optimizers = op
        self.steps = 0
        self.reset_count = 0
        self.next_start_step = 10
        self.multi_factor = 2
        self.total_epoch = 0
        if lr_scheduler == 'sgdr':
            self.update_lr = self.update_lr_SGDR
        elif lr_scheduler == 'cos':
            self.update_lr = self.update_lr_cosine
        elif lr_scheduler == 'zigzag':
            self.update_lr = self.update_lr_zigzag
        elif lr_scheduler == 'none':
            self.update_lr = self.no_update

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
    def no_update(self, base_lr):
        return base_lr

    def update_lr_SGDR(self, base_lr):
        end_lr = 1e-3 # 0.001
        total_T = self.total_epoch + 1
        if total_T >= self.next_start_step:
            self.steps = 0
            self.next_start_step *= self.multi_factor
        cur_T = self.steps + 1
        lr = end_lr + 1/2 * (base_lr - end_lr) * (1.0 + math.cos(math.pi*cur_T/total_T))
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        self.total_epoch += 1
        return lr

    def update_lr_zigzag(self, base_lr):
        warmup_steps = 50
        annealing_steps = 20
        end_lr = 1e-4
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps+1) / warmup_steps
        elif self.steps < warmup_steps+annealing_steps:
            step = self.steps - warmup_steps
            q = (annealing_steps - step) / annealing_steps
            lr = base_lr * q + end_lr * (1 - q)
        else:
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr

    def update_lr_cosine(self, base_lr):
        """ update the learning rate of all params according to warmup and cosine annealing """
        # 400, 1e-3
        warmup_steps = 10
        annealing_steps = 500
        end_lr = 1e-3
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps+1) / warmup_steps
        elif self.steps < warmup_steps+annealing_steps:
            step = self.steps - warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / annealing_steps))
            lr = base_lr * q + end_lr * (1 - q)
        else:
            # lr = base_lr * 0.001
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr

def get_logger(name):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if name is not None:
        fh = logging.FileHandler(f'{name}.log')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger

def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx