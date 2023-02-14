

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool


from molrep.models.base_model import BaseModel
from molrep.common.registry import registry

from molrep.processors.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


@registry.register_model("gin")
class GIN(BaseModel):
    """
    GIN is a model which contains a message passing network following by feed-forward layers.
    """

    MODEL_CONFIG_DICT = {
        "gin_default": "configs/models/gin_default.yaml",
    }

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, max_num_nodes=200):
        super(GIN, self).__init__()

        self.dim_features = dim_features
        self.dim_target = dim_target
        self.max_num_nodes = max_num_nodes

        self.dropout = model_configs['dropout']
        self.hidden_size = model_configs['hidden_size']
        self.num_layers = model_configs['num_layers']
        self.JK = model_configs['JK']

        if model_configs['aggregation'] == 'sum':
            self.pooling = global_add_pool
        elif model_configs['aggregation'] == 'mean':
            self.pooling = global_mean_pool
        elif model_configs['aggregation'] == "max":
            self.pooling = global_max_pool
        else:
            raise NotImplementedError("Aggregation not implemented.")

        self.node_encoder = AtomEncoder(self.hidden_size)

        self.gnn_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.gnn_convs.append(GINConv(self.hidden_size))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_size))

        if self.JK == "concat":
            self.fc = nn.Linear((self.num_layers + 1) * self.hidden_size,
                                               self.dim_target)
        else:
            self.fc = nn.Linear(self.hidden_size, self.dim_target)


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

    @classmethod
    def from_config(cls, cfg=None):
        model_configs = cfg.model_cfg
        dataset_configs = cfg.datasets_cfg

        dim_features = dataset_configs.get("dim_features", 0)
        dim_target = dataset_configs.get("dim_target", 1)

        model = cls(
            dim_features=dim_features,
            dim_target=dim_target,
            dataset_configs=dataset_configs,
            model_configs=model_configs,
        )
        return model

    def reset_parameters(self):
        for emb in self.node_encoder.atom_embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)

        for i in range(self.num_layers):
            self.gnn_convs[i].reset_parameters()
            self.batch_norms[i].reset_parameters()

        self.fc.reset_parameters()
    
    def featurize(self, data):
        batch_data = data["pygdata"]
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        x.requires_grad = True

        self.conv_acts = []
        self.conv_grads = []

        x = self.node_encoder(x)
        x_list = [x]
        for layer_i, conv in enumerate(self.gnn_convs[:-1]):
            with torch.enable_grad():
                x = conv(x, edge_index, edge_attr)
            x.register_hook(self.activations_hook)
            self.conv_acts.append(x)

            x = self.batch_norms[layer_i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_list.append(x)

        with torch.enable_grad():
            x = self.gnn_convs[-1](x, edge_index, edge_attr)
        x.register_hook(self.activations_hook)
        x_list.append(x)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            output = torch.cat(x_list, dim=1)
        elif self.JK == "last":
            output = x_list[-1]
        elif self.JK == "max":
            x_list = [x.unsqueeze_(0) for x in x_list]
            output = torch.max(torch.cat(x_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            x_list = [x.unsqueeze_(0) for x in x_list]
            output = torch.sum(torch.cat(x_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")

        output = F.dropout(output, self.dropout, training=self.training)
        output = self.pooling(output, batch)
        return output

    def forward(self, data):
        bsz = data["pygdata"].x.size(0)
        out = self.featurize(data)
        out = self.fc(out)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            out = self.sigmoid(out)
        if self.multiclass:
            out = out.reshape((bsz, -1, self.multiclass_num_classes)) # batch size x num targets x num classes per target
            out = self.multiclass_softmax(out) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        return out

    def get_batch_nums(self, data):
        data = data["pygdata"]
        batch_nodes = data.x.shape[0]
        batch_edges = data.edge_attr.shape[0]
        return batch_nodes, batch_edges

    def get_gap_activations(self, data):
        output = self.forward(data)
        output.backward(torch.ones_like(output))
        return self.conv_acts[-1], None

    def get_prediction_weights(self):
        w = self.fc.weight.t()
        return w[:, 0]

    def get_intermediate_activations_gradients(self, data):
        output = self.forward(data)
        output.backward(torch.ones_like(output))
        return self.conv_acts, self.conv_grads

    def activations_hook(self, grad):
        self.conv_grads.append(grad)

    def get_gradients(self, batch_data):
        data = batch_data["pygdata"]
        data.x.requires_grad_()
        data.x.retain_grad()
        data.edge_attr.requires_grad_()
        data.edge_attr.retain_grad()

        output = self.forward({"pygdata": data})
        output.backward(torch.ones_like(output))

        atom_grads = data.x.grad
        bond_grads = data.edge_attr.grad
        return data.x, atom_grads, data.edge_attr, bond_grads

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        aggr (str): aggreagation method

    See https://arxiv.org/abs/1810.00826 """
    def __init__(self, emb_dim):
        super(GINConv, self).__init__()

        # some implementations using batchnorm1d
        # see: https://github.com/PattanaikL/chiral_gnn/blob/master/model/layers.py#L74
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                                 nn.BatchNorm1d(2 * emb_dim),
                                 nn.ReLU(),
                                 nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = BondEncoder(emb_dim)

    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)

        for emb in self.edge_encoder.bond_embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x +
                       self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class AtomEncoder(nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i].long())
        return x_embedding


class BondEncoder(nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i].long())
        return bond_embedding