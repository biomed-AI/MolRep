import os
import copy
import scipy
import pickle

import numpy as np
from scipy.fft import dstn
import scipy.sparse as sp
import networkx as nx

import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix
from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from MolRep.Interactions.link_models.CFLP.utils import *


class CFLP(nn.Module):
    # def __init__(self, dim_feat, dim_h, dim_z, dropout, gnn_type='GCN', jk_mode='mean', dec='hadamard'):
    def __init__(self, model_configs, dataset_configs):
        super(CFLP, self).__init__()
        self.model_configs = model_configs
        self.dataset_configs = dataset_configs
        self.device = model_configs['device']
        self.eval_metric = "hits@20" if self.dataset_configs['eval_metric'] == 'hits' else self.dataset_configs['eval_metric']

        gcn_num_layers = 3
        self.encoder = GNN(dataset_configs['num_node_feats'], model_configs['dim_h'], model_configs['dim_z'], model_configs['dropout'], gnn_type=model_configs['gnn_type'], num_layers = gcn_num_layers, jk_mode=model_configs['jk_mode']).to(model_configs['device'])
        if model_configs['jk_mode'] == 'cat':
            dim_in = model_configs['dim_h'] * (gcn_num_layers-1) + model_configs['dim_z']
        else:
            dim_in = model_configs['dim_z']
        self.decoder = Decoder(model_configs['dec'], dim_in, dim_h=model_configs['dim_h']).to(model_configs['device'])
        self.init_params()

            
        # Parameters and Optimizer
        self.para_list = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optim = torch.optim.Adam(self.para_list,
                                 lr=self.model_configs['lr'],
                                 weight_decay=self.model_configs['l2reg'])
        self.optims = MultipleOptimizer(self.model_configs['lr_scheduler'], optim)

    def forward(self, adj, features, edges, T_f_batch, T_cf_batch):
        z = self.encoder(adj, features)
        z_i = z[edges.T[0]]
        z_j = z[edges.T[1]]
        logits_f = self.decoder(z_i, z_j, T_f_batch)
        logits_cf = self.decoder(z_i, z_j, T_cf_batch)
        return z, logits_f, logits_cf

    def init_params(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def processing_data(self, data, split_edge):
        ds = self.dataset_configs['name']
        path = self.dataset_configs['path']

        # adj_train = data.adjacency_matrix(scipy_fmt='csr')
        data_cp = copy.deepcopy(data)
        data_sparse = T.ToSparseTensor()(data_cp)
        row, col, _ = data_sparse.adj_t.coo()
        N = data_sparse.num_nodes
        
        if hasattr(data, 'edge_feat'):
            edge_attr = edge_attr.view(-1).cpu()
        else:
            edge_attr = torch.ones(row.size(0)) 
        assert edge_attr.size(0) == row.size(0)
        adj_train = scipy.sparse.coo_matrix(
                (edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N)).tocsr(copy=False)

        g = to_networkx(data)
        print('density',nx.density(g))
        print('edges:', len(g.edges()))
        print('nodes:', len(g.nodes()))
        adj_train.setdiag(1)
        if hasattr(data, 'x') and data.x is not None:
            features = data.x
        else:
            # construct one-hot degree features
            degrees = torch.LongTensor(adj_train.sum(0) - 1)
            indices = torch.cat((torch.arange(adj_train.shape[0]).unsqueeze(0), degrees), dim=0)
            features = torch.sparse.FloatTensor(indices, torch.ones(adj_train.shape[0])).to_dense().numpy()
            features = torch.Tensor(features)
        # using adj_train as adj_label as training loss is only calculated with train_pairs (excluding val/test edges and no_edges)
        adj_label = copy.deepcopy(adj_train)
        # load given train/val/test edges and no_edges
        # split_edge = dataset.get_edge_split()
        val_split, test_split = split_edge["valid"], split_edge["test"]
        val_edges, val_edges_false = val_split['edge'].numpy(), val_split['edge_neg'].numpy()
        test_edges, test_edges_false = test_split['edge'].numpy(), test_split['edge_neg'].numpy()
        # get training node pairs (edges and no-edges)
        if os.path.exists(f'{path}/{ds}_trainpairs.pkl'):
            train_pairs = pickle.load(open(f'{path}/{ds}_trainpairs.pkl', 'rb'))
        else:
            train_mask = np.ones(adj_train.shape)
            for edges_tmp in [val_edges, val_edges_false, test_edges, test_edges_false]:
                train_mask[edges_tmp.T[0], edges_tmp.T[1]] = 0
                train_mask[edges_tmp.T[1], edges_tmp.T[0]] = 0
            train_pairs = np.asarray(sp.triu(train_mask, 1).nonzero()).T
            pickle.dump(train_pairs, open(f'{path}/{ds}_trainpairs.pkl', 'wb'))


        # load n by n treatment matrix
        T_file_path = f'{path}/T_files/'
        if not os.path.exists(T_file_path):
            os.makedirs(T_file_path, exist_ok=True)
        T_file = f"{T_file_path}{ds}_{self.model_configs['t']}{self.model_configs['k']}-{self.model_configs['dist']}{self.model_configs['gamma']}-{self.model_configs['embraw']}.pkl"
        T_f, edges_cf_t1, edges_cf_t0, T_cf, adj_cf = load_t_files(self.model_configs, self.dataset_configs, T_file, adj_train)

        
        # get the factual node pairs
        edges_f_t1 = np.asarray((sp.triu(T_f, 1) > 0).nonzero()).T
        edges_f_t0 = np.asarray(sp.triu(T_f==0, 1).nonzero()).T
        assert edges_f_t1.shape[0] + edges_f_t0.shape[0] == np.arange(adj_label.shape[0]).sum()

        
        # get train_edges and train_edges_false
        # self.dataset_dir = self.dataset_config["path"]
        trainsplit_dir_name = f'{path}/train_split/'
        if not os.path.exists(trainsplit_dir_name):
            os.makedirs(trainsplit_dir_name, exist_ok=True)
        try:
            train_edges, train_edges_false = pickle.load(open(f'{trainsplit_dir_name}{ds}.pkl', 'rb'))
        except:
            train_edges = np.asarray(sp.triu(adj_train, 1).nonzero()).T
            all_set = set([tuple(x) for x in train_pairs])
            edge_set = set([tuple(x) for x in train_edges])
            noedge_set = all_set - edge_set
            train_edges_false = np.asarray(list(noedge_set))
            pickle.dump((train_edges, train_edges_false), open(f'{trainsplit_dir_name}{ds}.pkl', 'wb'))

        assert train_edges.shape[0] + train_edges_false.shape[0] == train_pairs.shape[0]


        max_neg_rate = train_edges_false.shape[0] // train_edges.shape[0] - 1
        if self.model_configs['neg_rate'] > max_neg_rate:
            self.model_configs['neg_rate'] = max_neg_rate
        val_pairs = np.concatenate((val_edges, val_edges_false), axis=0)
        val_labels = np.concatenate((np.ones(val_edges.shape[0]), np.zeros(val_edges_false.shape[0])), axis=0)
        test_pairs = np.concatenate((test_edges, test_edges_false), axis=0)
        test_labels = np.concatenate((np.ones(test_edges.shape[0]), np.zeros(test_edges_false.shape[0])), axis=0)

        # cast everything to proper type
        # adj_train_coo = data_sparse.adj_t.coo()
        # edge_index = np.concatenate((adj_train_coo.row[np.newaxis,:],adj_train_coo.col[np.newaxis,:]), axis=0)
        edge_index = torch.stack([col, row], dim=0)
        adj_norm = torch_sparse.SparseTensor.from_edge_index(torch.LongTensor(edge_index))

        # move everything to device
        device = self.device
        adj_norm = adj_norm.to(device)
        T_f = torch.FloatTensor(T_f.toarray()).to(device)
        T_cf = torch.FloatTensor(T_cf.toarray()).to(device)
        T_f_val = T_f[val_pairs.T]
        T_f_test = T_f[test_pairs.T]
        adj_cf = torch.FloatTensor(adj_cf.toarray()).to(device)
        features = features.to(device)
        pos_w_f = torch.FloatTensor([self.model_configs['neg_rate']]).to(device)

        return {
                'train_edges': train_edges, 'train_edges_false': train_edges_false, 'adj_cf': adj_cf, 'adj_norm': adj_norm, 
                'T_f':  T_f, 'T_cf': T_cf, 'T_f_val': T_f_val, 'T_f_test': T_f_test, 'edges_f_t1': edges_f_t1, 'edges_f_t0':edges_f_t0, 'edges_cf_t1':edges_cf_t1, 'edges_cf_t0':edges_cf_t0, 'pos_w_f':pos_w_f,
                'val_pairs': val_pairs, 'val_labels': val_labels, 'val_edges': val_labels,
                'test_pairs': test_pairs, 'test_labels': test_labels, 'test_edges': test_edges,
                'features': features,
        }


    def pretrain(self, inputs, logger=None):
        train_edges, train_edges_false = inputs['train_edges'], inputs['train_edges_false']
        adj_cf, T_f, T_cf, adj_norm, features = inputs['adj_cf'], inputs['T_f'], inputs['T_cf'], inputs['adj_norm'], inputs['features']
        edges_f_t1, edges_f_t0, edges_cf_t1, edges_cf_t0, pos_w_f = inputs['edges_f_t1'], inputs['edges_f_t0'], inputs['edges_cf_t1'], inputs['edges_cf_t0'], inputs['pos_w_f']
        
        val_pairs, val_labels, val_edges, T_f_val = inputs['val_pairs'], inputs['val_labels'], inputs['val_edges'], inputs['T_f_val']
        test_pairs, test_labels, test_edges, T_f_test = inputs['test_pairs'], inputs['test_labels'], inputs['test_edges'], inputs['T_f_test']

        batch_size, neg_rate, lr, num_np = self.model_configs['batch_size'], self.model_configs['neg_rate'], self.model_configs['lr'], self.model_configs['num_np']

        best_val_res = 0.0
        pretrained_params = None
        cnt_wait = 0
        for epoch in range(self.model_configs['pretrain_epochs']):
            
            total_loss = total_examples = 0
            for perm in DataLoader(range(train_edges.shape[0]), batch_size, shuffle=True):
                # sample no_edges for this batch
                pos_edges =  train_edges[perm]
                neg_sample_idx = np.random.choice(train_edges_false.shape[0], neg_rate * len(perm), replace=False)
                neg_edges = train_edges_false[neg_sample_idx]
                train_edges_batch = np.concatenate((pos_edges, neg_edges), axis=0)
                # move things to device
                labels_f_batch = torch.cat((torch.ones(pos_edges.shape[0]), torch.zeros(neg_edges.shape[0]))).to(self.device)
                labels_cf_batch = adj_cf[train_edges_batch.T]
                T_f_batch = T_f[train_edges_batch.T]
                T_cf_batch = T_cf[train_edges_batch.T]
                pos_w_cf = (labels_cf_batch.shape[0] - labels_cf_batch.sum()) / labels_cf_batch.sum()

                self.encoder.train()
                self.decoder.train()
                lr = self.optims.update_lr(lr)
                self.optims.zero_grad()
                # forward pass
                z, logits_f, logits_cf = self.forward(adj_norm, features, train_edges_batch, T_f_batch, T_cf_batch)
                # loss
                nodepairs_f, nodepairs_cf = sample_nodepairs(num_np, edges_f_t1, edges_f_t0, edges_cf_t1, edges_cf_t0)
                loss_disc = calc_disc(self.model_configs['disc_func'], z, nodepairs_f, nodepairs_cf)
                loss_f = F.binary_cross_entropy_with_logits(logits_f, labels_f_batch, pos_weight=pos_w_f)
                loss_cf = F.binary_cross_entropy_with_logits(logits_cf, labels_cf_batch, pos_weight=pos_w_cf)
                loss = loss_f + self.model_configs['alpha'] * loss_cf + self.model_configs['beta'] * loss_disc
                
                #backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
                self.optims.step()

                total_loss += loss.item() * pos_edges.shape[0]
                total_examples += pos_edges.shape[0]

            total_loss /= total_examples

            #evaluation
            self.encoder.eval()
            self.decoder.eval()

            with torch.no_grad():
                z = self.encoder(adj_norm, features)
                logits_val = self.decoder(z[val_pairs.T[0]], z[val_pairs.T[1]], T_f_val).detach().cpu()
                logits_test = self.decoder(z[test_pairs.T[0]], z[test_pairs.T[1]], T_f_test).detach().cpu()
            val_res = eval_ep_batched(logits_val, val_labels, val_edges.shape[0])
            if val_res[self.eval_metric] >= best_val_res:
                cnt_wait = 0
                best_val_res = val_res[self.eval_metric]
                pretrained_params = parameters_to_vector(self.para_list)
                test_res = eval_ep_batched(logits_test, test_labels, test_edges.shape[0])
                test_res['best_val'] = val_res[self.eval_metric]
                # logger.log('Epoch {} Loss: {:.4f} lr: {:.4f} val: {:.4f} test: {:.4f}'.format(
                #         epoch+1, total_loss, lr, val_res[self.eval_metric], test_res[self.eval_metric]))
            else:
                cnt_wait += 1
                # logger.log('Epoch {} Loss: {:.4f} lr: {:.4f} val: {:.4f}'.format(
                #         epoch+1, total_loss, lr, val_res[self.eval_metric]))

            if cnt_wait >= self.model_configs['patience']:
                print('Early stopping!')
                break

        return pretrained_params

    def obtained_z_from_encoder(self, inputs, pretrained_params):
        adj_norm, features = inputs['adj_norm'], inputs['features']
        vector_to_parameters(pretrained_params, self.para_list)

        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(adj_norm, features).detach()
        return z

    def train(self, inputs, other=None):
        z = other['z']

        train_edges, train_edges_false = inputs['train_edges'], inputs['train_edges_false']
        T_f, pos_w_f = inputs['T_f'], inputs['pos_w_f']
        
        batch_size, neg_rate, lr, num_np = self.model_configs['batch_size'], self.model_configs['neg_rate'], self.model_configs['lr'], self.model_configs['num_np']

        optim_ft = torch.optim.Adam(self.decoder.parameters(),
                                    lr=self.model_configs['lr_ft'],
                                    weight_decay=self.model_configs['l2reg'])
        
        total_loss = total_examples = 0
        # sample no_edges for this epoch
        for perm in DataLoader(range(train_edges.shape[0]), batch_size, shuffle=True):
            pos_edges =  train_edges[perm]
            neg_sample_idx = np.random.choice(train_edges_false.shape[0], neg_rate * len(perm), replace=False)
            neg_edges = train_edges_false[neg_sample_idx]
            train_edges_batch = np.concatenate((pos_edges, neg_edges), axis=0)
            labels_f_batch = torch.cat((torch.ones(pos_edges.shape[0]), torch.zeros(neg_edges.shape[0]))).to(self.device)
            T_f_batch = T_f[train_edges_batch.T]

            self.decoder.train()
            optim_ft.zero_grad()

            logits = self.decoder(z[train_edges_batch.T[0]], z[train_edges_batch.T[1]], T_f_batch)
            loss = F.binary_cross_entropy_with_logits(logits, labels_f_batch, pos_weight=pos_w_f)

            loss.backward()
            optim_ft.step()

            total_loss += loss.item() * pos_edges.shape[0]
            total_examples += pos_edges.shape[0]
        
        total_loss /= total_examples
        return total_loss

    def test(self, inputs, evaluator, other=None):
        z = other['z']
        
        val_pairs, val_labels, val_edges, T_f_val = inputs['val_pairs'], inputs['val_labels'], inputs['val_edges'], inputs['T_f_val']
        test_pairs, test_labels, test_edges, T_f_test = inputs['test_pairs'], inputs['test_labels'], inputs['test_edges'], inputs['T_f_test']

        self.decoder.eval()
        with torch.no_grad():
            logits_val = self.decoder(z[val_pairs.T[0]], z[val_pairs.T[1]], T_f_val).detach().cpu()
            logits_test = self.decoder(z[test_pairs.T[0]], z[test_pairs.T[1]], T_f_test).detach().cpu()

        val_res = eval_ep_batched(logits_val, val_labels, val_edges.shape[0], evaluator=evaluator)
        test_res = eval_ep_batched(logits_test, test_labels, test_edges.shape[0], evaluator=evaluator)

        return 0.0, 0.0, (val_res, test_res)



class GNN(nn.Module):
    def __init__(self, dim_feat, dim_h, dim_z, dropout, gnn_type='GCN', num_layers=3, jk_mode='mean', batchnorm=True):
        super(GNN, self).__init__()

        assert jk_mode in ['max','sum','mean','lstm','cat','none']
        self.act = nn.ELU()
        self.dropout = dropout
        self.linear = torch.nn.Linear(dim_h, dim_z)

        if gnn_type == 'SAGE':
            gnnlayer = SAGEConv
        elif gnn_type == 'GCN':
            gnnlayer = GCNConv
        self.convs = torch.nn.ModuleList()
        self.convs.append(gnnlayer(dim_feat, dim_h))
        for _ in range(num_layers - 2):
            self.convs.append(gnnlayer(dim_h, dim_h))
        self.convs.append(gnnlayer(dim_h, dim_z))

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim_h) for _ in range(num_layers)])

        self.jk_mode = jk_mode
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk = JumpingKnowledge(mode=self.jk_mode, channels=dim_h, num_layers=num_layers)
        elif self.jk_mode == 'mean':
            self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def forward(self, adj, features):
        out = features
        out_list = []

        for i in range(len(self.convs)):
            out = self.convs[i](out, adj)
            if self.batchnorm:
                 out = self.bns[i](out)
            out = self.act(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            out_list += [out]

        if self.jk_mode in ['max', 'lstm', 'cat']:
            out = self.jk(out_list)
        elif self.jk_mode == 'mean':
            sftmax = F.softmax(self.weights, dim=0)
            for i in range(len(out_list)):
                out_list[i] = out_list[i] * sftmax[i]
                out = sum(out_list)
        elif self.jk_mode == 'sum':
            out_stack = torch.stack(out_list, dim=0)
            out = torch.sum(out_stack, dim=0)
        elif self.jk_mode == 'none':
            out = out_list[-1]
        return out

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
        if self.jk_mode in ['max', 'lstm', 'cat']:
            self.jk.reset_parameters()


class Decoder(nn.Module):
    def __init__(self, dec, dim_z, dim_h=64):
        super(Decoder, self).__init__()
        self.dec = dec
        if dec == 'innerproduct':
            dim_in = 2
        elif dec == 'hadamard':
            dim_in = dim_z + 1
        elif dec == 'mlp':
            dim_in = 1 + 2*dim_z
        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, dim_h, bias=True),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(dim_h, 1, bias=False)
        )

    def forward(self, z_i, z_j, T):
        if self.dec == 'innerproduct':
            z = (z_i * z_j).sum(1).view(-1, 1)
            h = torch.cat((z, T.view(-1, 1)), dim=1)
        elif self.dec == 'mlp':
            h = torch.cat((z_i, z_j, T.view(-1, 1)), dim=1)
        elif self.dec == 'hadamard':
            z = z_i * z_j
            h = torch.cat((z, T.view(-1, 1)), dim=1)
        h = self.mlp_out(h).squeeze()
        return h

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue