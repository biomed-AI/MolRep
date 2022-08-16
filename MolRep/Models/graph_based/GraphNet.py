
from operator import mod
from numpy.lib.arraysetops import isin
import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import NNConv, Set2Set


class GraphNet(nn.Module):
    def __init__(self, dim_features, dim_target, model_configs, dataset_configs):
        super().__init__()

        assert isinstance(dim_features, tuple)
        node_features, edge_features = dim_features

        dim_node_hidden = model_configs['dim_node_hidden']
        dim_edge_hidden = model_configs['dim_edge_hidden']
        num_layers = model_configs['num_layers']
        num_step_set2set = model_configs['num_step_set2set']
        num_layer_set2set = model_configs['num_layer_set2set']
        aggr_type = model_configs['aggregation_type']

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_features, dim_node_hidden),
            nn.ReLU()
        )

        self.num_layers = num_layers

        fnet = nn.Sequential(
            nn.Linear(edge_features, dim_edge_hidden),
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


    def featurize(self, data):
        x, edge_index, edge_attrs, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.project_node_feats(x)     # (batch_size, node hidden features)
        hidden_feats = x.unsqueeze(0)      # (1, batch_size, node hidden features)

        for _ in range(self.num_layers):

            x = self.gnn(x, edge_index, edge_attrs)

            x = F.relu(x)
            x, hidden_feats = self.gru(x.unsqueeze(0), hidden_feats)
            x = x.squeeze(0)

        graph_feats = self.readout(x, batch)
        return graph_feats

    def forward(self, data):
        x, edge_index, edge_attrs, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x.requires_grad = True
        edge_attrs.requires_grad = True

        x = self.project_node_feats(x)     # (batch_size, node hidden features)
        hidden_feats = x.unsqueeze(0)      # (1, batch_size, node hidden features)

        self.conv_acts = []
        self.conv_grads = []

        for _ in range(self.num_layers):

            with torch.enable_grad():
                x = self.gnn(x, edge_index, edge_attrs)

            x.register_hook(self.activations_hook)
            self.conv_acts.append(x)

            x = F.relu(x)
            x, hidden_feats = self.gru(x.unsqueeze(0), hidden_feats)
            x = x.squeeze(0)

        graph_feats = self.readout(x, batch)

        x = F.relu(self.fc1(graph_feats))
        x = self.fc2(x)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            x = self.sigmoid(x)
        if self.multiclass:
            x = x.reshape(x.size(0), -1, self.multiclass_num_classes) # batch size x num targets x num classes per target
            # if not self.training:
            x = self.multiclass_softmax(x) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return x

    def get_gap_activations(self, data):
        output = self.forward(data)
        output.backward(torch.ones_like(output))
        return self.conv_acts[-1], None

    def get_prediction_weights(self):
        w = self.fc2.weight.t()
        return w[:, 0]

    def get_intermediate_activations_gradients(self, data):
        output = self.forward(data)
        output.backward(torch.ones_like(output))
        return self.conv_acts, self.conv_grads

    def activations_hook(self, grad):
        self.conv_grads.append(grad)

    def get_gradients(self, data):
        data.x.requires_grad_()
        data.x.retain_grad()

        data.edge_attr.requires_grad_()
        data.edge_attr.retain_grad()

        output = self.forward(data)
        output.backward(torch.ones_like(output))

        atom_grads = data.x.grad
        bond_grads = data.edge_attr.grad
        return data.x, atom_grads, data.edge_attr, bond_grads