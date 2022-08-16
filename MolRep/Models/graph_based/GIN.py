import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GIN(torch.nn.Module):

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, max_num_nodes=200):
        super(GIN, self).__init__()

        self.dropout = model_configs['dropout']
        self.embeddings_dim = [model_configs['hidden_size']] + [model_configs['hidden_size']] * model_configs['num_layers']
        self.max_num_nodes = max_num_nodes
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = model_configs['train_eps']
        if model_configs['aggregation'] == 'sum':
            self.pooling = global_add_pool
        elif model_configs['aggregation'] == 'mean':
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

        self.task_type = dataset_configs["task_type"]
        self.multiclass_num_classes = dataset_configs["multiclass_num_classes"] if self.task_type == 'Multi-Classification' else None

        self.classification = self.task_type == 'Classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = self.task_type == 'Multi-Classification'
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        self.regression = self.task_type == 'Regression'
        if self.regression:
            self.relu = nn.ReLU()
        assert not (self.classification and self.regression and self.multiclass)

    def unbatch(self, x, batch):
        sizes = degree(batch, dtype=torch.long).tolist()
        node_feat_list = x.split(sizes, dim=0)

        feat = torch.zeros((len(node_feat_list), self.max_num_nodes, self.embeddings_dim)).to(x.device)
        mask = torch.ones((len(node_feat_list), self.max_num_nodes)).to(x.device)

        for idx, node_feat in enumerate(node_feat_list):
            node_num = node_feat.size(0)
            
            if node_num <= self.max_num_nodes:
                feat[idx, :node_num] = node_feat
                mask[idx, node_num:] = 0
            
            else:
                feat[idx] = node_feat[:self.max_num_nodes]

        return feat, mask

    def featurize(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.linears[layer](x), batch, p=self.dropout)
            elif layer == self.no_layers - 1:
                x = self.convs[layer-1](x, edge_index)
                out = x
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                out += F.dropout(self.linears[layer](x), p=self.dropout, training=self.training)
       
        return self.unbatch(out, batch)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            out = self.sigmoid(out)
        if self.multiclass:
            out = out.reshape((x.size(0), -1, self.multiclass_num_classes)) # batch size x num targets x num classes per target
            # if not self.training:
            out = self.multiclass_softmax(out) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return out