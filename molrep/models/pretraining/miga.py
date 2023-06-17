#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
@time:     2023-06-10
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

from molrep.common.registry import registry
from molrep.common.config import Config
from molrep.models.base_model import BaseModel
from molrep.models.pretraining.miga_utils import calc_contrastive_loss, do_AttrMasking, do_InfoGraph, do_ContextPred


@registry.register_model("miga")
class MIGA(BaseModel):
    """
    MIGA: 
    """

    MODEL_CONFIG_DICT = {
        "default": "configs/models/miga.yaml",
    }

    def __init__(self, model_configs, gnn_model_configs):
        super(MIGA, self).__init__()
        
        self.molecule_gnn_name = model_configs.gnn.model.name
        self.molecule_gnn_model_configs = gnn_model_configs.model
        self.molecule_gnn_model = GNN(self.molecule_gnn_model_configs.num_layer,
                                      self.molecule_gnn_model_configs.emb_dim,
                                      JK = self.molecule_gnn_model_configs.JK,
                                      drop_ratio = self.molecule_gnn_model_configs.dropout_ratio,
                                      gnn_type = self.molecule_gnn_model_configs.gnn_type)
        self.molecule_readout_func = global_mean_pool

        self.molecule_cnn_model = Cnn(model_name=model_configs.cnn_model_name, target_num = model_configs.emb_dim, n_ch=3 if model_configs.cDNA else 5, pretrained= True)
        self.project_layer1 = nn.Sequential(nn.Linear(model_configs.emb_dim, model_configs.emb_dim), nn.LeakyReLU(), nn.BatchNorm1d(model_configs.emb_dim), nn.Linear(model_configs.emb_dim, model_configs.emb_dim))
        self.project_layer2 = nn.Sequential(nn.Linear(model_configs.emb_dim, model_configs.emb_dim), nn.LeakyReLU(), nn.BatchNorm1d(model_configs.emb_dim), nn.Linear(model_configs.emb_dim, model_configs.emb_dim))
        self.gim_head = nn.Linear(self.model_configs.emb_dim*2, 2)

        self.molecule_img_generator = VariationalAutoEncoder(
                emb_dim=model_configs.emb_dim, loss=model_configs.generator_loss, detach_target=model_configs.detach_target,
                beta=model_configs.beta)
        self.molecule_graph_generator = VariationalAutoEncoder(
                emb_dim=model_configs.emb_dim, loss=model_configs.generator_loss, detach_target=model_configs.detach_target,
                beta=model_configs.beta)

        # set up Graph SSL model
        self.MGM_models = []
        self.MGM_mode = model_configs.MGM_mode

        # set up Graph SSL model for IG
        infograph_discriminator_SSL_model = None
        if model_configs.MGM_mode == 'IG':
            infograph_discriminator_SSL_model = Discriminator(model_configs.emb_dim)
            self.MGM_models.append(infograph_discriminator_SSL_model)

        # set up Graph SSL model for AM
        molecule_atom_masking_model = None
        if model_configs.MGM_mode == 'AM':
            molecule_atom_masking_model = torch.nn.Linear(model_configs.emb_dim, 119)
            self.MGM_models.append(molecule_atom_masking_model)

        # set up Graph SSL model for CP
        molecule_context_model = None
        if model_configs.MGM_mode == 'CP':
            l1 = model_configs.num_layer - 1
            l2 = l1 + model_configs.csize
            molecule_context_model = GNN(int(l2 - l1), self.molecule_gnn_model_configs.emb_dim, JK=self.molecule_gnn_model_configs.JK,
                                        drop_ratio=self.molecule_gnn_model_configs.dropout_ratio, gnn_type=self.molecule_gnn_model_configs.gnn_type)
            self.MGM_models.append(molecule_context_model)

        if model_configs.MGM_mode == 'MGM':
            molecule_atom_masking_model = torch.nn.Linear(model_configs.emb_dim, 119)
            self.MGM_models.append(molecule_atom_masking_model)

            l1 = model_configs.num_layer - 1
            l2 = l1 + model_configs.csize
            molecule_context_model = GNN(int(l2 - l1), self.molecule_gnn_model_configs.emb_dim, JK=self.molecule_gnn_model_configs.JK,
                                        drop_ratio=self.molecule_gnn_model_configs.dropout_ratio, gnn_type=self.molecule_gnn_model_configs.gnn_type)
            self.MGM_models.append(molecule_context_model)

    def forward(self, data):
        if isinstance(data, dict):
            data = data["pygdata"]
        
        batch1, batch2 = data
        batch1 = batch1.to(self.device)
        batch2 = batch2.to(self.device)

        node_repr = self.molecule_gnn_model(batch1.x, batch1.edge_index, batch1.edge_attr)
        molecule_graph_emb = self.molecule_readout_func(node_repr, batch1.batch)
        molecule_img_emb = self.molecule_cnn_model(batch2)

        molecule_graph_emb = self.project_layer1(molecule_graph_emb)
        molecule_img_emb = self.project_layer2(molecule_img_emb)

        ##### To obtain Graph-Image SSL loss and acc
        GIC_loss = calc_contrastive_loss(molecule_graph_emb, molecule_img_emb, args) #INFO or EBM

        img_generator_loss = self.molecule_img_generator(molecule_graph_emb, molecule_img_emb)
        graph_generator_loss = self.molecule_graph_generator(molecule_img_emb, molecule_graph_emb)
        GIM_loss = (img_generator_loss + graph_generator_loss) / 2

        # Graph-Image Matching Loss
        # forward the positve image-text pair
        output_pos = torch.cat([molecule_graph_emb, molecule_img_emb], dim=1)

        with torch.no_grad():
            bs = molecule_graph_emb.size(0)
            weights_g2i = F.softmax(torch.cdist(molecule_graph_emb, molecule_img_emb, p=2))
            weights_i2g = F.softmax(torch.cdist(molecule_img_emb, molecule_graph_emb, p=2))
            weights_i2g.fill_diagonal_(0)
            weights_g2i.fill_diagonal_(0)

        # select a negative image for each text
        molecule_img_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_g2i[b], 1).item()
            molecule_img_embeds_neg.append(molecule_img_emb[neg_idx])
        molecule_img_embeds_neg = torch.stack(molecule_img_embeds_neg,dim=0)

        # select a negative text for each image
        molecule_graph_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2g[b], 1).item()
            molecule_graph_embeds_neg.append(molecule_graph_emb[neg_idx])
        molecule_graph_embeds_neg = torch.stack(molecule_graph_embeds_neg,dim=0)

        molecule_graph_embeds_all = torch.cat([molecule_graph_emb, molecule_graph_embeds_neg],dim=0)
        molecule_img_embeds_all = torch.cat([molecule_img_emb,molecule_img_embeds_neg],dim=0)

        output_neg = torch.cat([molecule_graph_embeds_all, molecule_img_embeds_all], dim=1)

        vl_embeddings = torch.cat([output_pos, output_neg],dim=0)
        vl_output = self.gim_head(vl_embeddings)

        gim_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                            dim=0).to(molecule_graph_emb.device)
        GGIM_loss = GIM_loss + F.cross_entropy(vl_output, gim_labels)


        if self.MGM_mode == 'IG':
            criterion = nn.BCEWithLogitsLoss()
            infograph_discriminator_SSL_model = self.MGM_models[0]
            MGM_loss, _ = do_InfoGraph(
                node_repr=node_repr, batch=batch1,
                molecule_repr=molecule_graph_emb, criterion=criterion,
                infograph_discriminator_SSL_model=infograph_discriminator_SSL_model)

        elif self.MGM_mode == 'AM':
            criterion = nn.CrossEntropyLoss()
            molecule_atom_masking_model = self.MGM_models[0]
            masked_node_repr = self.molecule_gnn_model(batch1.masked_x, batch1.edge_index, batch1.edge_attr)
            MGM_loss, _ = do_AttrMasking(
                batch=batch1, criterion=criterion, node_repr=masked_node_repr,
                molecule_atom_masking_model=molecule_atom_masking_model)

        elif self.MGM_mode == 'CP':
            criterion = nn.BCEWithLogitsLoss()
            molecule_context_model = self.MGM_models[0]
            MGM_loss, _ = do_ContextPred(
                batch=batch1, criterion=criterion,
                molecule_substruct_model=self.molecule_gnn_model,
                molecule_context_model=molecule_context_model,
                molecule_readout_func=self.molecule_readout_func
                )

        elif self.MGM_mode == 'MGM':

            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.BCEWithLogitsLoss()
            criterion = [criterion1, criterion2]

            molecule_atom_masking_model = self.MGM_models[0]
            masked_node_repr = self.molecule_gnn_model(batch1.masked_x, batch1.edge_index, batch1.edge_attr)
            MGM_loss1, _ = do_AttrMasking(
                batch=batch1, criterion=criterion[0], node_repr=masked_node_repr,
                molecule_atom_masking_model=molecule_atom_masking_model)

            molecule_context_model = self.MGM_models[1]
            MGM_loss2, _ = do_ContextPred(
                batch=batch1, criterion=criterion[1],
                molecule_substruct_model=self.molecule_gnn_model,
                molecule_context_model=molecule_context_model,
                molecule_readout_func=self.molecule_readout_func,
                molecule_img_repr=molecule_img_emb
                )

            MGM_loss = MGM_loss1 + MGM_loss2
        else:
            raise Exception

        return {
            'loss': GIC_loss + GGIM_loss + MGM_loss,
            'GIC_loss': GIC_loss, 'GGIM_loss': GGIM_loss,
            'MGM_loss': MGM_loss,
        }


num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add", task_type='classification'):
        super(GINConv, self).__init__()
        self.task_type = task_type
        if self.task_type == 'classification':
            #multi-layer perceptron
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
            self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
            self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

            torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
            self.aggr = aggr
        else:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
            self.eps = torch.nn.Parameter(torch.Tensor([0]))

            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
            

    def forward(self, x, edge_index, edge_attr):

        if self.task_type == 'classification':
            #add self loops in the edge space
            edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
            edge_index = edge_index[0]
            self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
            return self.propagate(edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)
        else:
            edge_embedding = self.bond_encoder(edge_attr)
            out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
            return out

    def message(self, x_j, edge_attr):
        # return F.relu(x_j + edge_attr)
        if self.task_type == 'classification':
            return x_j + edge_attr
        else:
            return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)
    


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin", task_type = 'classification'):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.task_type = task_type
        if self.task_type == 'classification':
            self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.atom_encoder = AtomEncoder(emb_dim)


        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if self.task_type == 'classification':
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim=1)


class Cnn(nn.Module):
    def __init__(self, model_name, target_num, n_ch=5, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained)
        # print(self.model)
        if ('efficientnet' in model_name) or ('mixnet' in model_name):
            self.model.conv_stem.weight = nn.Parameter(self.model.conv_stem.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif model_name in ['resnet34d']:
            self.model.conv1[0].weight = nn.Parameter(self.model.conv1[0].weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif ('resnet' in model_name or 'resnest' in model_name) and 'vit' not in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif 'rexnet' in model_name or 'regnety' in model_name or 'nf_regnet' in model_name:
            self.model.stem.conv.weight = nn.Parameter(self.model.stem.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.fc.in_features, target_num)
            self.model.head.fc = nn.Identity()
        elif 'resnext' in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif 'hrnet_w32' in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif 'densenet' in model_name:
            self.model.features.conv0.weight = nn.Parameter(self.model.features.conv0.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif 'ese_vovnet39b' in model_name or 'xception41' in model_name:
            self.model.stem[0].conv.weight = nn.Parameter(self.model.stem[0].conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.fc.in_features, target_num)
            self.model.head.fc = nn.Identity()
        elif 'dpn' in model_name:
            self.model.features.conv1_1.conv.weight = nn.Parameter(self.model.features.conv1_1.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_channels, target_num)
            self.model.classifier = nn.Identity()
        elif 'inception' in model_name:
            self.model.features[0].conv.weight = nn.Parameter(self.model.features[0].conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.last_linear.in_features, target_num)
            self.model.last_linear = nn.Identity()
        elif 'vit' in model_name:
            self.model.patch_embed.proj.weight = nn.Parameter(self.model.patch_embed.proj.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.in_features, target_num)
            self.model.head = nn.Identity()
        elif 'vit_base_resnet50' in model_name:
            self.model.patch_embed.backbone.stem.conv.weight = nn.Parameter(self.model.patch_embed.backbone.stem.conv.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.head.in_features, target_num)
            self.model.head = nn.Identity()
        else:
            raise

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

    def extract(self, x):
        return self.model(x)

    def forward(self, x):
        x = self.extract(x)
        return self.myfc(x)
    

def cosine_similarity(p, z, average=True):
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    loss = -(p * z).sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1):
        super(VariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'cosine':
            self.criterion = cosine_similarity

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = reconstruction_loss + self.beta * kl_loss

        return loss