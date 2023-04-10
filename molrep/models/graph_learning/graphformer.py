


import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool

from molrep.models.base_model import BaseModel, AtomEncoder, BondEncoder, ModelOutputs
from molrep.common.registry import registry


@registry.register_model("graphformer")
class Graphformer(BaseModel):
    """
    Graphformer is a model which contains a message passing network following by feed-forward layers.
    """

    MODEL_CONFIG_DICT = {
        "default": "configs/models/graphformer_default.yaml",
    }

    def __init__(self, dim_target, model_configs):
        super(Graphformer, self).__init__()
        self.dim_target = dim_target

        self.n_layers = model_configs['n_layers']
        self.num_heads = model_configs['num_heads']
        self.hidden_dim = model_configs['hidden_dim']
        self.dropout_rate = model_configs['dropout_rate']
        self.intput_dropout_rate = model_configs['intput_dropout_rate']
        self.ffn_dim = model_configs['ffn_dim']
        self.attention_dropout_rate = model_configs['attention_dropout_rate']
        self.multi_hop_max_dist = model_configs['multi_hop_max_dist']

        self.atom_encoder = nn.Embedding(512 * 9 + 1, self.hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(512 * 3 + 1, self.num_heads, padding_idx=0)
        self.edge_type = model_configs['edge_type']
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(128 * self.num_heads * self.num_heads, 1)
        self.rel_pos_encoder = nn.Embedding(512, self.num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(self.intput_dropout_rate)
        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(self.hidden_dim)

        self.downstream_out_proj = nn.Linear(self.hidden_dim, self.dim_target)

        self.graph_token = nn.Embedding(1, self.hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, self.num_heads)

    @classmethod
    def from_config(cls, cfg=None):
        model_configs = cfg.model_cfg
        dataset_configs = cfg.datasets_cfg

        dim_target = dataset_configs.get("dim_target", 1)

        model = cls(
            dim_target=dim_target,
            model_configs=model_configs,
        )
        return model

    def forward(self, data):
        if isinstance(data, dict):
            data = data["pygdata"]

        attn_bias, rel_pos, x = data.attn_bias, data.rel_pos, data.x
        in_degree, out_degree = data.in_degree, data.in_degree
        edge_input, attn_edge_type = data.edge_input, data.attn_edge_type

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # rel pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                        :, 1:, 1:] + rel_pos_bias  # spatial encoder
        # reset rel pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            rel_pos_ = rel_pos.clone()
            rel_pos_[rel_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_)
            if self.multi_hop_max_dist > 0:
                rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(
                3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) /
                          (rel_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(
                attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input  # edge encoder
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)           # [n_graph, n_node, n_hidden]

        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)  # degree encoder
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        output = self.downstream_out_proj(output[:, 0, :])  # virtual node

        return ModelOutputs(
            logits=output,
        )


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x