import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import SAGEConv, global_max_pool

import torch
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, configs=None):
        super().__init__()

        num_layers = model_configs['num_layers']
        dim_embedding = model_configs['dim_embedding']
        self.aggregation = model_configs['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

        self.task_type = dataset_configs["task_type"]
        self.multiclass_num_classes = dataset_configs["multiclass_num_classes"] if self.task_type == 'Multi-Classification' else None

        self.classification = self.task_type == 'Classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = self.task_type == 'Multiclass-Classification'
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        self.regression = self.task_type == 'Regression'
        if self.regression:
            self.relu = nn.ReLU()
        assert not (self.classification and self.regression and self.multiclass)

    def featurize(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)
        x = torch.cat(x_all, dim=1)

        return x

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            x = self.sigmoid(x)
        if self.multiclass:
            x = x.reshape((x.size(0), -1, self.multiclass_num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                x = self.multiclass_softmax(x) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return x
