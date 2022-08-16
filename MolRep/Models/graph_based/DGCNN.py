import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_sort_pool
from torch_geometric.utils import add_self_loops, degree


class DGCNN(nn.Module):
    """
    Uses fixed architecture
    """

    def __init__(self, dim_features, dim_target, model_configs, dataset_configs, max_num_nodes=200):
        super(DGCNN, self).__init__()

        self.ks = {"0.6": 79 , "0.9": 120}

        self.k = self.ks[str(model_configs['k'])] 
        self.embedding_dim = model_configs['embedding_dim']
        self.num_layers = model_configs['num_layers']
        self.max_num_nodes = max_num_nodes

        self.convs = []
        for layer in range(self.num_layers):
            input_dim = dim_features if layer == 0 else self.embedding_dim
            self.convs.append(DGCNNConv(input_dim, self.embedding_dim))
        self.total_latent_dim = self.num_layers * self.embedding_dim

        # Add last embedding
        self.convs.append(DGCNNConv(self.embedding_dim, 1))
        self.total_latent_dim += 1

        self.convs = nn.ModuleList(self.convs)
        self.conv1d_params1 = nn.Conv1d(1, 16, self.total_latent_dim, self.total_latent_dim)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(16, 32, 5, 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        self.input_dense_dim = (dense_dim - 5 + 1) * 32

        self.hidden_dense_dim = model_configs['dense_dim']
        self.dense_layer = nn.Sequential(nn.Linear(self.input_dense_dim, self.hidden_dense_dim),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(self.hidden_dense_dim, dim_target))


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

        feat = torch.zeros((len(node_feat_list), self.max_num_nodes, self.embedding_dim)).to(x.device)
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

        hidden_repres = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            hidden_repres.append(x)
        # x = torch.cat(hidden_repres, dim=1)
        x = torch.stack(hidden_repres, dim=1).mean(1)
        return self.unbatch(x, batch)

    def forward(self, data):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x.requires_grad = True

        self.conv_acts = []
        self.conv_grads = []
        hidden_repres = []

        for conv in self.convs:
            with torch.enable_grad():
                x = conv(x, edge_index)
            x.register_hook(self.activations_hook)
            self.conv_acts.append(x)
            x = torch.tanh(x)
            hidden_repres.append(x)

        # apply sortpool
        x_to_sortpool = torch.cat(hidden_repres, dim=1)
        x_1d = global_sort_pool(x_to_sortpool, batch, self.k)  # in the code the authors sort the last channel only
        # apply 1D convolutional layers
        x_1d = torch.unsqueeze(x_1d, dim=1) 
        conv1d_res = F.relu(self.conv1d_params1(x_1d))
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = F.relu(self.conv1d_params2(conv1d_res))
        conv1d_res = conv1d_res.reshape(conv1d_res.shape[0], -1) 

        # apply dense layer
        out_dense = self.dense_layer(conv1d_res)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification:# and not self.training:
            out_dense = self.sigmoid(out_dense)
        if self.multiclass:
            out_dense = out_dense.reshape((out_dense.size(0), -1, self.multiclass_num_classes)) # batch size x num targets x num classes per target
            # if not self.training:
            out_dense = self.multiclass_softmax(out_dense) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return out_dense

    
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

        conv_grads = [conv_g.grad for conv_g in self.conv_grads]
        return self.conv_acts, self.conv_grads

    def activations_hook(self, grad):
        self.conv_grads.append(grad)

    def get_gradients(self, data):
        data.x.requires_grad_()
        data.x.retain_grad()
        output = self.forward(data)
        output.backward()

        atom_grads = data.x.grad
        return data.x, atom_grads, None, None


class DGCNNConv(MessagePassing):
    """
    Extended from tuorial on GCNs of Pytorch Geometrics
    """

    def __init__(self, in_channels, out_channels):
        super(DGCNNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print("in DGCNN unit:",edge_index.shape)
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x ,e=edge_index,s = (x.size(0), x.size(0)))

    def message(self, x_j, e, s):#r fixed  
        # x_j has shape [E, out_channels]
        edge_index = e
        size = s
        # Step 3: Normalize node features.
        src, dst = edge_index  # we assume source_to_target message passing
        deg = degree(src, size[0], dtype=x_j.dtype)
        deg = deg.pow(-1)
        norm = deg[dst]

        return norm.view(-1, 1) * x_j  # broadcasting the normalization term to all out_channels === hidden features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)