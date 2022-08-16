import math

import torch.nn as nn
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce, SparseTensor

from MolRep.Interactions.link_models.PLNLP.loss import *
from MolRep.Interactions.link_models.PLNLP.utils import *
from MolRep.Interactions.link_models.PLNLP.layers import *


class PLNLP(nn.Module):
    """
        Parameters
        ----------
        lr : double
            Learning rate
        dropout : double
            dropout probability for gnn and mlp layers
        gnn_num_layers : int
            number of gnn layers
        mlp_num_layers : int
            number of gnn layers
        *_hidden_channels : int
            dimension of hidden
        num_nodes : int
            number of graph nodes
        num_node_feats : int
            dimension of raw node features
        gnn_encoder_name : str
            gnn encoder name
        predictor_name: str
            link predictor name
        loss_func: str
            loss function name
        optimizer_name: str
            optimization method name
        device: str
            device name: gpu or cpu
        use_node_feats: bool
            whether to use raw node features as input
        train_node_emb: bool
            whether to train node embeddings based on node id
        pretrain_emb: str
            whether to load pretrained node embeddings
    """

    # def __init__(self, lr, dropout, grad_clip_norm, gnn_num_layers, mlp_num_layers, emb_hidden_channels,
    #              gnn_hidden_channels, mlp_hidden_channels, num_nodes, num_node_feats, gnn_encoder_name,
    #              predictor_name, loss_func, optimizer_name, device, use_node_feats, train_node_emb,
    #              pretrain_emb=None):
    def __init__(self, model_configs, dataset_configs):
        super(PLNLP, self).__init__()

        self.model_configs = model_configs
        self.dataset_configs = dataset_configs

        self.loss_func_name = model_configs['loss_func']
        self.num_nodes = dataset_configs['num_nodes']
        self.num_node_feats = dataset_configs['num_node_feats']
        self.use_node_feats = model_configs['use_node_feats']
        self.train_node_emb = model_configs['train_node_emb']
        self.clip_norm = model_configs['grad_clip_norm']
        self.device = model_configs['device']

        # Input Layer
        self.input_channels, self.emb = create_input_layer(num_nodes=dataset_configs['num_nodes'],
                                                           num_node_feats=dataset_configs['num_node_feats'],
                                                           hidden_channels=model_configs['emb_hidden_channels'],
                                                           use_node_feats=model_configs['use_node_feats'],
                                                           train_node_emb=model_configs['train_node_emb'],
                                                           pretrain_emb=model_configs['pretrain_emb'])
        if self.emb is not None:
            self.emb = self.emb.to(model_configs['device'])

        # GNN Layer
        self.encoder = create_gnn_layer(input_channels=self.input_channels,
                                        hidden_channels=model_configs['gnn_hidden_channels'],
                                        num_layers=model_configs['gnn_num_layers'],
                                        dropout=model_configs['dropout'],
                                        encoder_name=model_configs['gnn_encoder_name']).to(model_configs['device'])

        # Predict Layer
        self.predictor = create_predictor_layer(hidden_channels=model_configs['mlp_hidden_channels'],
                                                num_layers=model_configs['mlp_num_layers'],
                                                dropout=model_configs['dropout'],
                                                predictor_name=model_configs['predictor_name']).to(model_configs['device'])

        # Parameters and Optimizer
        self.para_list = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.emb is not None:
            self.para_list += list(self.emb.parameters())

        if model_configs['optimizer_name'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.para_list, lr=model_configs['lr'])
        elif model_configs['optimizer_name'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.para_list, lr=model_configs['lr'], momentum=0.9, weight_decay=1e-5, nesterov=True)
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=model_configs['lr'])

        self.param_init()

    def param_init(self):
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        if self.emb is not None:
            torch.nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, data):
        if self.use_node_feats:
            input_feat = data.x.to(self.device)
            if self.train_node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def calculate_loss(self, pos_out, neg_out, num_neg, margin=None):
        if self.loss_func_name == 'CE':
            loss = ce_loss(pos_out, neg_out)
        elif self.loss_func_name == 'InfoNCE':
            loss = info_nce_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'LogRank':
            loss = log_rank_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'HingeAUC':
            loss = hinge_auc_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'AdaAUC' and margin is not None:
            loss = adaptive_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'WeightedAUC' and margin is not None:
            loss = weighted_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'AdaHingeAUC' and margin is not None:
            loss = adaptive_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        elif self.loss_func_name == 'WeightedHingeAUC' and margin is not None:
            loss = weighted_hinge_auc_loss(pos_out, neg_out, num_neg, margin)
        else:
            loss = auc_loss(pos_out, neg_out, num_neg)
        return loss

    def processing_data(self, data, split_edge):
        
        data = T.ToSparseTensor()(data).to(self.device)
        row, col, _ = data.adj_t.coo()
        data.edge_index = torch.stack([col, row], dim=0)
        

        if self.dataset_configs['name'].replace('_', '-') == 'ogbl-citation2':
            data.adj_t = data.adj_t.to_symmetric()
        
        if self.dataset_configs['name'].replace('_', '-') == 'ogbl-collab':
            # only train edges after specific year
            if self.dataset_configs['year'] > 0 and hasattr(data, 'edge_year'):
                selected_year_index = torch.reshape(
                    (split_edge['train']['year'] >= self.dataset_configs['year']).nonzero(as_tuple=False), (-1,))
                split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
                split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
                split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
                train_edge_index = split_edge['train']['edge'].t()
                # create adjacency matrix
                new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
                new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
                data.adj_t = SparseTensor(row=new_edge_index[0],
                                        col=new_edge_index[1],
                                        value=new_edge_weight.to(torch.float32))
                data.edge_index = new_edge_index

            # Use training + validation edges
            if self.dataset_configs['use_valedges_as_input']:
                full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
                full_edge_weight = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=-1)
                # create adjacency matrix
                new_edges = to_undirected(full_edge_index, full_edge_weight, reduce='add')
                new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
                data.adj_t = SparseTensor(row=new_edge_index[0],
                                        col=new_edge_index[1],
                                        value=new_edge_weight.to(torch.float32))
                data.edge_index = new_edge_index

                if self.dataset_configs['use_coalesce']:
                    full_edge_index, full_edge_weight = coalesce(full_edge_index, full_edge_weight, self.dataset_configs['num_nodes'], self.dataset_configs['num_nodes'])

                # edge weight normalization
                split_edge['train']['edge'] = full_edge_index.t()
                deg = data.adj_t.sum(dim=1).to(torch.float)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                split_edge['train']['weight'] = deg_inv_sqrt[full_edge_index[0]] * full_edge_weight * deg_inv_sqrt[
                    full_edge_index[1]]

        if self.model_configs['gnn_encoder_name'].upper() == 'GCN':
            # Pre-compute GCN normalization.
            data.adj_t = gcn_normalization(data.adj_t)

        if self.model_configs['gnn_encoder_name'].upper() == 'WSAGE':
            data.adj_t = adj_normalization(data.adj_t)

        if self.model_configs['gnn_encoder_name'].upper() == 'TRANSFORMER':
            row, col, edge_weight = data.adj_t.coo()
            data.adj_t = SparseTensor(row=row, col=col)

        return {'data': data, 'split_edge': split_edge}

    def train(self, inputs, other=None):
        # data, split_edge = self.processing_data(data, split_edge)
        data, split_edge = inputs['data'], inputs['split_edge']
        batch_size, neg_sampler_name, num_neg = self.model_configs['batch_size'], self.model_configs['neg_sampler_name'], self.model_configs['num_neg']

        self.encoder.train()
        self.predictor.train()

        pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge,
                                                           edge_index=data.edge_index,
                                                           num_nodes=self.num_nodes,
                                                           neg_sampler_name=neg_sampler_name,
                                                           num_neg=num_neg)

        pos_train_edge, neg_train_edge = pos_train_edge.to(self.device), neg_train_edge.to(self.device)

        if 'weight' in split_edge['train']:
            edge_weight_margin = split_edge['train']['weight'].to(self.device)
        else:
            edge_weight_margin = None

        total_loss = total_examples = 0

        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad()

            input_feat = self.create_input_feat(data)
            h = self.encoder(input_feat, data.adj_t)
            pos_edge = pos_train_edge[perm].t()
            neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2)).t()

            pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]])
            neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])

            weight_margin = edge_weight_margin[perm] if edge_weight_margin is not None else None
            loss = self.calculate_loss(pos_out, neg_out, num_neg, margin=weight_margin)
            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)

            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def batch_predict(self, h, edges, batch_size):
        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            edge = edges[perm].t()
            preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, inputs, batch_size, evaluator, eval_metric):
        # data, split_edge = self.processing_data(data, split_edge)
        data, split_edge = inputs['data'], inputs['split_edge']
    
        self.encoder.eval()
        self.predictor.eval()

        input_feat = self.create_input_feat(data)
        h = self.encoder(input_feat, data.adj_t)
        # The default index of unseen nodes is -1,
        # hidden representations of unseen nodes is the average of all seen node representations
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h = torch.cat([h, mean_h], dim=0)

        pos_valid_edge, neg_valid_edge = get_pos_neg_edges('valid', split_edge)
        pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge)
        pos_valid_edge, neg_valid_edge = pos_valid_edge.to(self.device), neg_valid_edge.to(self.device)
        pos_test_edge, neg_test_edge = pos_test_edge.to(self.device), neg_test_edge.to(self.device)

        pos_valid_pred = self.batch_predict(h, pos_valid_edge, batch_size)
        neg_valid_pred = self.batch_predict(h, neg_valid_edge, batch_size)

        h = self.encoder(input_feat, data.adj_t)
        mean_h = torch.mean(h, dim=0, keepdim=True)
        h = torch.cat([h, mean_h], dim=0)

        pos_test_pred = self.batch_predict(h, pos_test_edge, batch_size)
        neg_test_pred = self.batch_predict(h, neg_test_edge, batch_size)

        if eval_metric.startswith('hits'):
            val_results = evaluate_hits(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
            )
            test_results = evaluate_hits(
                evaluator,
                pos_test_pred,
                neg_test_pred
            )
        else:
            val_results = evaluate_mrr(
                evaluator,
                pos_valid_pred,
                neg_valid_pred)
            test_results = evaluate_mrr(
                evaluator,
                pos_test_pred,
                neg_test_pred)

        return 0, 0, (val_results, test_results)


def create_input_layer(num_nodes, num_node_feats, hidden_channels, use_node_feats=True,
                       train_node_emb=False, pretrain_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if train_node_emb:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim += hidden_channels
        elif pretrain_emb is not None and pretrain_emb != '':
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim += emb.weight.size(1)
    else:
        if pretrain_emb is not None and pretrain_emb != '':
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim = emb.weight.size(1)
        else:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim = hidden_channels
    return input_dim, emb


def create_gnn_layer(input_channels, hidden_channels, num_layers, dropout=0, encoder_name='SAGE'):
    if encoder_name.upper() == 'GCN':
        return GCN(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'WSAGE':
        return WSAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'TRANSFORMER':
        return Transformer(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    else:
        return SAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)


def create_predictor_layer(hidden_channels, num_layers, dropout=0, predictor_name='MLP'):
    predictor_name = predictor_name.upper()
    if predictor_name == 'DOT':
        return DotPredictor()
    elif predictor_name == 'BIL':
        return BilinearPredictor(hidden_channels)
    elif predictor_name == 'MLP':
        return MLPPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPDOT':
        return MLPDotPredictor(hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPBIL':
        return MLPBilPredictor(hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPCAT':
        return MLPCatPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)


def adjust_lr(optimizer, decay_ratio, lr):
    lr_ = lr * (1 - decay_ratio)
    lr_min = lr * 0.0001
    if lr_ < lr_min:
        lr_ = lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_