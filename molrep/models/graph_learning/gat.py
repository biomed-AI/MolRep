
import os
import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.utils import degree
from torch_geometric.nn import GATConv, global_max_pool

from molrep.models.base_model import BaseModel, AtomEncoder, ModelOutputs
from molrep.common.registry import registry


@registry.register_model("gat")
class GAT(BaseModel):
    """
    GAT is a model which contains a message passing network following by feed-forward layers.
    """

    MODEL_CONFIG_DICT = {
        "default": "configs/models/gat_default.yaml",
    }

    def __init__(self, dim_features, dim_target, model_configs, max_num_nodes=200):
        super().__init__()

        num_layers = model_configs['num_layers']
        dim_embedding = model_configs['dim_embedding']
        heads = model_configs['head']
        dropout = model_configs['dropout']
        self.aggregation = model_configs['aggregation']  # can be mean or max

        self.dim_embedding = dim_embedding
        self.max_num_nodes = max_num_nodes

        self.node_encoder = AtomEncoder(dim_features)

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding*heads

            conv = GATConv(dim_input, dim_embedding, heads=heads, dropout=dropout)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(heads * num_layers * dim_embedding, heads * dim_embedding)
        self.fc2 = nn.Linear(heads * dim_embedding, dim_embedding)
        self.fc3 = nn.Linear(dim_embedding, dim_target)

    @classmethod
    def from_config(cls, cfg=None):
        model_configs = cfg.model_cfg
        dataset_configs = cfg.datasets_cfg

        dim_features = dataset_configs.get("dim_features", 0)
        dim_target = dataset_configs.get("dim_target", 1)

        model = cls(
            dim_features=dim_features,
            dim_target=dim_target,
            model_configs=model_configs,
        )
        return model

    def forward(self, data):
        batch_data = data["pygdata"]
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        node_feats = self.node_encoder(x)

        node_feats.requires_grad_()
        node_feats.retain_grad()

        x_all = []
        self.conv_acts = []
        self.conv_grads = []
        self.attention_weights = []

        x = node_feats
        for i, layer in enumerate(self.layers):

            with torch.enable_grad():
                x, attention_w = layer(x, edge_index, return_attention_weights=True)
            x.register_hook(self.activations_hook)
            self.conv_acts.append(torch.relu(x))
            self.attention_weights.append(attention_w)

            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return ModelOutputs(
            logits=x,
            node_features=node_feats, # (batch_size, node hidden features)
        )

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

    def featurize(self, data):
        batch_data = data["pygdata"]
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        
        x_all = []
        for i, layer in enumerate(self.layers):

            x, attention_w = layer(x, edge_index, return_attention_weights=True)
            
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)
        
        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def return_attention(self):
        return self.attention_weights

    def get_gap_activations(self, data):
        output = self.forward(data).logits
        output.backward(torch.ones_like(output))
        return self.conv_acts[-1], None

    def get_prediction_weights(self):
        w = self.fc2.weight.t()
        return w[:, 0]

    def get_intermediate_activations_gradients(self, data):
        output = self.forward(data).logits
        output.backward(torch.ones_like(output))
        return self.conv_acts, self.conv_grads

    def activations_hook(self, grad):
        self.conv_grads.append(grad)

    def get_gradients(self, data):
        if isinstance(data, dict):
            data = data["pygdata"]

        outputs = self.forward(data)
        logits = outputs.logits
        logits.backward(torch.ones_like(logits))

        atom_grads = outputs.node_features.grad
        return outputs.node_features, atom_grads, None, None