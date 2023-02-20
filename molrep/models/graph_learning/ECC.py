import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree, dense_to_sparse
from torch_geometric.nn import ECConv
from torch_scatter import scatter_add

from molrep.models.base_model import BaseModel
from molrep.common.registry import registry


def _make_block_diag(mats, mat_sizes):
    block_diag = torch.zeros(sum(mat_sizes), sum(mat_sizes))

    for i, (mat, size) in enumerate(zip(mats, mat_sizes)):
        cum_size = sum(mat_sizes[:i])
        block_diag[cum_size:cum_size+size,cum_size:cum_size+size] = mat
    return block_diag

class ECCLayer(nn.Module):
    def __init__(self, dim_input, dim_embedding, dropout=0.):
        super().__init__()

        fnet1 = nn.Sequential(nn.Linear(1, 16),
                              nn.ReLU(),
                              nn.Linear(16, dim_embedding * dim_input))

        fnet2 = nn.Sequential(nn.Linear(1, 16),
                              nn.ReLU(),
                              nn.Linear(16, dim_embedding * dim_embedding))

        fnet3 = nn.Sequential(nn.Linear(1, 16),
                              nn.ReLU(),
                              nn.Linear(16, dim_embedding * dim_embedding))

        self.conv1 = ECConv(dim_input, dim_embedding, nn=fnet1)
        self.conv2 = ECConv(dim_embedding, dim_embedding, nn=fnet2)
        self.conv3 = ECConv(dim_embedding, dim_embedding, nn=fnet3)

        self.bn1 = nn.BatchNorm1d(dim_embedding)
        self.bn2 = nn.BatchNorm1d(dim_embedding)
        self.bn3 = nn.BatchNorm1d(dim_embedding)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(self.bn1(x), p=self.dropout, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(self.bn2(x), p=self.dropout, training=self.training)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.dropout(self.bn3(x), p=self.dropout, training=self.training)

        return x


@registry.register_model("ecc")
class ECC(BaseModel):
    """
    Uses fixed architecture.

    IMPORTANT NOTE: we will consider dataset which do not have edge labels.
    Therefore, we avoid learning the function that associates a weight matrix
    to an edge specific weight.

    """

    MODEL_CONFIG_DICT = {
        "ecc_default": "configs/models/ecc_default.yaml",
    }

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, max_num_nodes=200):
        super().__init__()
        self.model_configs = model_configs
        self.dropout = model_configs['dropout']
        self.dropout_final = model_configs['dropout_final']
        self.num_layers = model_configs['num_layers']
        dim_embedding = model_configs['dim_embedding']

        self.dim_embedding = dim_embedding
        self.max_num_nodes = max_num_nodes

        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            dim_input = dim_features if i == 0 else dim_embedding
            layer = ECCLayer(dim_input, dim_embedding, dropout=self.dropout)
            self.layers.append(layer)

        fnet = nn.Sequential(nn.Linear(1, 16),
                             nn.ReLU(),
                             nn.Linear(16, dim_embedding * dim_embedding))

        self.final_conv = ECConv(dim_embedding, dim_embedding, nn=fnet)
        self.final_conv_bn = nn.BatchNorm1d(dim_embedding)

        self.fc1 = nn.Linear(dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)


        self.task_type = dataset_configs["task_type"]
        self.multiclass_num_classes = dataset_configs["multiclass_num_classes"] if self.task_type == 'multiclass-classification' else None

        self.classification = self.task_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = self.task_type == 'multiclass-classification'
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        self.regression = self.task_type == 'regression'
        if self.regression:
            self.relu = nn.ReLU()
        assert not (self.classification and self.regression and self.multiclass)

    @property
    def device(self):
        return list(self.parameters())[0].device

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

    @classmethod
    def default_config_path(cls, model_type):
        return os.path.join(registry.get_path("library_root"), cls.MODEL_CONFIG_DICT[model_type])

    def make_block_diag(self, matrix_list):
        mat_sizes = [m.size(0) for m in matrix_list]
        return _make_block_diag(matrix_list, mat_sizes)

    def get_ecc_conv_parameters(self, data, layer_no):
        v_plus_list, laplacians = data.v_plus, data.laplacians

        # print([v_plus[layer_no] for v_plus in v_plus_list])
        v_plus_batch = torch.cat([v_plus[layer_no] for v_plus in v_plus_list], dim=0)

        laplacian_layer_list = [laplacians[i][layer_no] for i in range(len(laplacians))]
        laplacian_block_diagonal = self.make_block_diag(laplacian_layer_list)

        # First layer
        lap_edge_idx, lap_edge_weights = dense_to_sparse(laplacian_block_diagonal)
        lap_edge_weights = lap_edge_weights.squeeze(-1)

        # Convert v_plus_batch to boolean
        return lap_edge_idx, lap_edge_weights, (v_plus_batch == 1)

    def unbatch(self, x, batch):
        sizes = degree(batch, dtype=torch.long).tolist()
        node_feat_list = x.split(sizes, dim=0)

        feat = torch.zeros((len(node_feat_list), self.max_num_nodes, self.dim_embedding)).to(x.device)
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
        data = data["pygdata"]
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, layer in enumerate(self.layers):
            # TODO should lap_edge_index[0] be equal to edge_idx?
            lap_edge_idx, lap_edge_weights, v_plus_batch = self.get_ecc_conv_parameters(data, layer_no=i)
            edge_index = lap_edge_idx if i != 0 else edge_index
            edge_weight = lap_edge_weights if i != 0 else x.new_ones((edge_index.size(1), ))

            edge_index = edge_index.to(self.model_configs["device"])
            edge_weight = edge_weight.to(self.model_configs["device"])

            x = layer(x, edge_index, edge_weight)

            # pooling
            x = x[v_plus_batch]
            batch = batch[v_plus_batch]

        # final_convolution
        lap_edge_idx, lap_edge_weight, v_plus_batch = self.get_ecc_conv_parameters(data, layer_no=self.num_layers)

        lap_edge_idx = lap_edge_idx.to(self.model_configs["device"])
        lap_edge_weight = lap_edge_weight.to(self.model_configs["device"])

        x = F.relu(self.final_conv(x, lap_edge_idx, lap_edge_weight.unsqueeze(-1)))
        
        return self.unbatch(x, batch)

    def forward(self, data):
        data = data["pygdata"]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x.requires_grad = True

        self.conv_acts = []
        self.conv_grads = []
        self.edge_grads = []

        for i, layer in enumerate(self.layers):
            # TODO should lap_edge_index[0] be equal to edge_idx?
            lap_edge_idx, lap_edge_weights, v_plus_batch = self.get_ecc_conv_parameters(data, layer_no=i)
            edge_index = lap_edge_idx if i != 0 else edge_index
            edge_weight = lap_edge_weights if i != 0 else x.new_ones((edge_index.size(1), ))

            edge_index = edge_index.to(self.model_configs["device"])
            edge_weight = edge_weight.to(self.model_configs["device"])
            edge_weight.requires_grad = True

            # apply convolutional layer
            with torch.enable_grad():
                x = layer(x, edge_index, edge_weight)
            x.register_hook(self.activations_hook)
            self.conv_acts.append(x)

            edge_weight.register_hook(self.edge_attrs_hook)

            # pooling
            x = x[v_plus_batch]
            batch = batch[v_plus_batch]

        # final_convolution
        lap_edge_idx, lap_edge_weight, v_plus_batch = self.get_ecc_conv_parameters(data, layer_no=self.num_layers)

        lap_edge_idx = lap_edge_idx.to(self.model_configs["device"])
        lap_edge_weight = lap_edge_weight.to(self.model_configs["device"])

        lap_edge_weight.requires_grad = True
        x = F.relu(self.final_conv(x, lap_edge_idx, lap_edge_weight.unsqueeze(-1)))
        x = F.dropout(self.final_conv_bn(x), p=self.dropout, training=self.training)

        lap_edge_weight.register_hook(self.edge_attrs_hook)
        self.lap_edge_weight = lap_edge_weight

        # TODO: is the following line needed before global pooling?
        # batch = batch[v_plus_batch]

        graph_emb = global_mean_pool(x, batch)

        x = F.relu(self.fc1(graph_emb))
        x = F.dropout(x, p=self.dropout_final, training=self.training)

        # No ReLU specified here todo check with source code (code is not so clear)
        x = self.fc2(x)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            x = self.sigmoid(x)
        if self.multiclass:
            x = x.reshape((x.size(0), -1, self.multiclass_num_classes)) # batch size x num targets x num classes per target
            # if not self.training:
            x = self.multiclass_softmax(x) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        return x

    def get_node_feats(self, data):
        data = data["pygdata"]
        return data.x.shape

    def get_edge_feats(self, data):
        data = data["pygdata"]
        return data.edge_attr.shape

    def get_gap_activations(self, data):
        output = self.forward(data)
        output.backward()
        return self.conv_acts[-1], None

    def get_prediction_weights(self):
        w = self.fc2.weight.t()
        return w[:, 0]

    def get_intermediate_activations_gradients(self, data):
        output = self.forward(data)
        output.backward()
        return self.conv_acts, self.conv_grads

    def activations_hook(self, grad):
        self.conv_grads.append(grad)

    def edge_attrs_hook(self, grad):
        self.edge_grads.append(grad)

    def get_gradients(self, batch_data):
        data = batch_data["pygdata"]
        data.x.requires_grad_()
        data.x.retain_grad()
        output = self.forward({"pygdata": data})
        output.backward()

        atom_grads = data.x.grad
        edge_grads_list = [edge_g.grad for edge_g in self.edge_grads]
        edge_grads = edge_grads_list[-1]
        return data.x, atom_grads, self.lap_edge_weight, edge_grads