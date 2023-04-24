

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool

from molrep.models.base_model import BaseModel, AtomEncoder, BondEncoder, ModelOutputs
from molrep.common.registry import registry


@registry.register_model("gin")
class GIN(BaseModel):
    """
    GIN is a model which contains a message passing network following by feed-forward layers.
    """

    MODEL_CONFIG_DICT = {
        "default": "configs/models/gin_default.yaml",
    }

    def __init__(self, model_configs, dim_target, max_num_nodes=200):
        super(GIN, self).__init__()
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
        self.edge_encoder = BondEncoder(self.hidden_size)

        self.gnn_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            self.gnn_convs.append(GINConv(self.hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_size))

        if dim_target is not None and self.JK == "concat":
            self.fc = nn.Linear((self.num_layers + 1) * self.hidden_size,
                                 self.dim_target)
        elif dim_target is not None:
            self.fc = nn.Linear(self.hidden_size, self.dim_target)

    @classmethod
    def from_config(cls, cfg=None):
        model_configs = cfg.model_cfg
        dataset_configs = cfg.datasets_cfg

        dim_target = dataset_configs.get("dim_target", 1)

        model = cls(
            dim_target=dim_target,
            model_configs=model_configs,
        )
        return model
    
    @classmethod
    def from_pretrained(cls, model_type):
        return super().from_pretrained(model_type)

    def reset_parameters(self):
        for emb in self.node_encoder.atom_embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)

        for emb in self.edge_encoder.bond_embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)

        for i in range(self.num_layers):
            self.gnn_convs[i].reset_parameters()
            self.batch_norms[i].reset_parameters()
        self.fc.reset_parameters()

    def forward(self, data):
        if isinstance(data, dict):
            data = data["pygdata"]

        x, edge_index, edge_attrs, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_feats = self.node_encoder(x)
        edge_feats = self.edge_encoder(edge_attrs)

        node_feats.requires_grad_()
        node_feats.retain_grad()
        edge_feats.requires_grad_()
        edge_feats.retain_grad()

        self.conv_acts = []
        self.conv_grads = []

        x = node_feats
        x_list = [x]
        for layer_i, conv in enumerate(self.gnn_convs[:-1]):
            with torch.enable_grad():
                x = conv(x, edge_index, edge_feats)

            x.register_hook(self.activations_hook)
            self.conv_acts.append(x)

            x = self.batch_norms[layer_i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_list.append(x)

        with torch.enable_grad():
            x = self.gnn_convs[-1](x, edge_index, edge_feats)
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
        output = self.fc(output)

        return ModelOutputs(
            logits=output,
            node_features=node_feats, # (batch_size * node_num, hidden features)
            edge_features=edge_feats, # (batch_size * edge_num, hidden features)
        )

    def get_node_representation(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attrs = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            if isinstance(data, dict):
                data = data["pygdata"]
            x, edge_index, edge_attrs = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        node_feats = self.node_encoder(x)
        edge_feats = self.edge_encoder(edge_attrs)

        x = node_feats
        x_list = [x]
        for layer_i, conv in enumerate(self.gnn_convs[:-1]):
            x = conv(x, edge_index, edge_feats)
            x = self.batch_norms[layer_i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_list.append(x)

        x = self.gnn_convs[-1](x, edge_index, edge_feats)
        x_list.append(x)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(x_list, dim=1)
        elif self.JK == "last":
            node_representation = x_list[-1]
        elif self.JK == "max":
            x_list = [x.unsqueeze_(0) for x in x_list]
            node_representation = torch.max(torch.cat(x_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            x_list = [x.unsqueeze_(0) for x in x_list]
            node_representation = torch.sum(torch.cat(x_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation

    def featurize(self, data):
        node_representation = self.get_node_representation(data)
        graph_output = F.dropout(node_representation, self.dropout, training=self.training)
        graph_output = self.pooling(graph_output, data.batch)
        return graph_output

    def unbatch(self, x, data, is_atom=True):
        if isinstance(data, dict):
            data = data["pygdata"]
        batch = data.batch

        if is_atom:
            sizes = degree(batch, dtype=torch.int64).tolist()
            return list(x.split(sizes, dim=0))
        else:
            edge_batch = batch[data.edge_index[0]]
            sizes = degree(edge_batch, dtype=torch.int64).tolist()
            return list(x.split(sizes, dim=0))

    def get_gap_activations(self, data):
        output = self.forward(data).logits
        output.backward(torch.ones_like(output))
        return self.conv_acts[-1], None

    def get_prediction_weights(self):
        w = self.fc.weight.t()
        return w[:, 0]

    def get_intermediate_activations_gradients(self, data):
        output = self.forward(data).logits
        output.backward(torch.ones_like(output))
        return self.conv_acts, self.conv_grads

    def activations_hook(self, grad):
        self.conv_grads.append(grad)

    def savegrads_hook(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def get_gradients(self, data):
        if isinstance(data, dict):
            data = data["pygdata"]

        outputs = self.forward(data)
        logits = outputs.logits
        logits.backward(torch.ones_like(logits))

        atom_grads = outputs.node_features.grad
        bond_grads = outputs.edge_features.grad
        return outputs.node_features, atom_grads, outputs.edge_features, bond_grads


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

    def reset_parameters(self):
        for c in self.mlp.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.eps.data, 0)

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) * x +
                       self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

