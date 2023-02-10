
import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.utils import degree
from torch_geometric.nn import GATConv, global_max_pool

class GAT(nn.Module):
    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, max_num_nodes=200):
        super().__init__()

        num_layers = model_configs['num_layers']
        dim_embedding = model_configs['dim_embedding']
        heads = model_configs['head']
        dropout = model_configs['dropout']
        self.aggregation = model_configs['aggregation']  # can be mean or max

        self.dim_embedding = dim_embedding
        self.max_num_nodes = max_num_nodes

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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x_all = []
        for i, layer in enumerate(self.layers):

            x, attention_w = layer(x, edge_index, return_attention_weights=True)
            
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)
        x = torch.stack(x_all, dim=1).mean(1)
        return self.unbatch(x, batch)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x.requires_grad = True

        x_all = []
        self.conv_acts = []
        self.conv_grads = []
        self.attention_weights = []

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

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            x = self.sigmoid(x)
        if self.multiclass:
            x = x.reshape((x.size(0), -1, self.multiclass_num_classes)) # batch size x num targets x num classes per target
            # if not self.training:
            x = self.multiclass_softmax(x) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return x

    def return_attention(self):
        return self.attention_weights

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
        output = self.forward(data)
        output.backward(torch.ones_like(output))

        atom_grads = data.x.grad
        return data.x, atom_grads, None, None
