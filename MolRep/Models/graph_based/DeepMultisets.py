import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_add_pool


class DeepMultisets(torch.nn.Module):

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, configs=None):
        super(DeepMultisets, self).__init__()

        hidden_units = model_configs['hidden_units']

        self.fc_vertex = Linear(dim_features, hidden_units)
        self.fc_global1 = Linear(hidden_units, hidden_units)
        self.fc_global2 = Linear(hidden_units, dim_target)


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


    def forward(self, data):
        x, batch = data.x, data.batch

        x = F.relu(self.fc_vertex(x))
        x = global_add_pool(x, batch)  # sums all vertex embeddings belonging to the same graph!
        x = F.relu(self.fc_global1(x))
        x = self.fc_global2(x)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            x = self.sigmoid(x)
        if self.multiclass:
            x = x.reshape((x.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                x = self.multiclass_softmax(x) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return x