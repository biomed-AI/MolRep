
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import degree

from torch_geometric.nn import NNConv, Set2Set
from molrep.models.base_model import BaseModel, AtomEncoder, BondEncoder, ModelOutputs
from molrep.common.registry import registry


@registry.register_model("graphnet")
class GraphNet(BaseModel):
    """
    GraphNet is a model which contains a message passing network following by feed-forward layers.
    """

    MODEL_CONFIG_DICT = {
        "default": "configs/models/graphnet_default.yaml",
    }

    def __init__(self, dim_target, model_configs, max_num_nodes=200):
        super().__init__()

        dim_node_hidden = model_configs['dim_node_hidden']
        dim_edge_hidden = model_configs['dim_edge_hidden']
        num_step_set2set = model_configs['num_step_set2set']
        num_layer_set2set = model_configs['num_layer_set2set']
        aggr_type = model_configs['aggregation_type']
        self.num_layers = model_configs['num_layers']
        self.max_num_nodes = max_num_nodes

        self.node_encoder = AtomEncoder(dim_node_hidden)
        self.edge_encoder = BondEncoder(dim_edge_hidden)

        self.project_node_feats = nn.Sequential(
            nn.Linear(dim_node_hidden, dim_node_hidden),
            nn.ReLU()
        )

        fnet = nn.Sequential(
            nn.Linear(dim_edge_hidden, dim_edge_hidden),
            nn.ReLU(),
            nn.Linear(dim_edge_hidden, dim_node_hidden * dim_node_hidden)
        )

        self.gnn = NNConv(
            in_channels=dim_node_hidden,
            out_channels=dim_node_hidden,
            nn=fnet,
            aggr=aggr_type
        )

        self.gru = nn.GRU(dim_node_hidden, dim_node_hidden)

        self.readout = Set2Set(
            in_channels=dim_node_hidden,
            processing_steps=num_step_set2set,
            num_layers=num_layer_set2set
        )

        # For graph classification
        self.fc1 = nn.Linear(2 * dim_node_hidden, dim_node_hidden)
        self.fc2 = nn.Linear(dim_node_hidden, dim_target)

    @classmethod
    def from_config(cls, cfg=None):
        model_configs = cfg.model_cfg
        dataset_configs = cfg.datasets_cfg

        dim_target = dataset_configs.get("dim_target", 1)
        max_num_nodes = dataset_configs.get("max_num_nodes", 200)

        model = cls(
            dim_target=dim_target,
            model_configs=model_configs,
            max_num_nodes=max_num_nodes,
        )
        return model

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

        x = self.project_node_feats(node_feats)     # (batch_size, node hidden features)
        hidden_feats = x.unsqueeze(0)      # (1, batch_size, node hidden features)

        self.conv_acts = []
        self.conv_grads = []

        for _ in range(self.num_layers):
            with torch.enable_grad():
                x = self.gnn(x, edge_index, edge_feats)

            x.register_hook(self.activations_hook)
            self.conv_acts.append(x)

            x = F.relu(x)
            x, hidden_feats = self.gru(x.unsqueeze(0), hidden_feats)
            x = x.squeeze(0)

        graph_feats = self.readout(x, batch)

        out = F.relu(self.fc1(graph_feats))
        out = self.fc2(out)

        return ModelOutputs(
            logits=out,
            node_features=node_feats, # (batch_size, node hidden features)
            edge_features=edge_feats, # (batch_size, edge hidden features)
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
        bond_grads = outputs.edge_features.grad
        return outputs.node_features, atom_grads, outputs.edge_features, bond_grads