
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool


class MorganFP(torch.nn.Module):

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs):
        super(MorganFP, self).__init__()
        hidden_dim = model_configs['hidden_units']
        dim_features = 2048
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_features, hidden_dim), nn.ReLU(),
                                       torch.nn.Linear(hidden_dim, dim_target))

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

    def forward(self, data):
        # print(data.x.shape)
        # print(data.x[:,:50])
        x = self.mlp(data.morgan_fp)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            x = self.sigmoid(x)
        if self.multiclass:
            x = x.reshape((x.size(0), -1, self.multiclass_num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                x = self.multiclass_softmax(x) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return x