

import torch
import torch.nn as nn

from molrep.common.registry import registry
from molrep.common.config import Config
from molrep.models.base_model import BaseModel


@registry.register_model("edge_pred")
class EdgePred(BaseModel):
    """
    Attribution Masking
    """
    MODEL_TYPES_DICT = {
        "pretraining": ("default")
    }

    MODEL_CONFIG_DICT = {
        "default": "configs/models/edge_pred.yaml",
    }

    def __init__(self, model_configs, gnn_model_configs):
        super(EdgePred, self).__init__()

        self.molecule_gnn_type = model_configs.gnn['gnn_type']
        self.molecule_gnn_model_configs = gnn_model_configs
        gnn_model_cls = registry.get_model_class(self.molecule_gnn_type)
        self.molecule_model = gnn_model_cls.from_config(
            model_configs=self.molecule_gnn_model_configs,
        )

        self.criterion = nn.BCEWithLogitsLoss()

    @classmethod
    def from_config(cls, cfg):
        model_configs = cfg.model_cfg
        gnn_model_configs = Config.build_model_config(model_configs.gnn)

        model = cls(
            model_configs=model_configs,
            gnn_model_configs=gnn_model_configs,
        )
        return model

    def forward(self, data):
        if isinstance(data, dict):
            data = data["pygdata"]

        node_repr = self.molecule_model(data.x, data.edge_index, data.edge_attr)

        edgepred_loss, edgepred_acc = self.do_EdgePred(
            node_repr=node_repr, batch=data)
        
        return {
            'edgepred_loss': edgepred_loss,
            'edgepred_acc': edgepred_acc,
        }

    def do_EdgePred(self, node_repr, batch):

        # positive/negative scores -> inner product of node features
        positive_score = torch.sum(node_repr[batch.edge_index[0, ::2]] *
                                node_repr[batch.edge_index[1, ::2]], dim=1)
        negative_score = torch.sum(node_repr[batch.negative_edge_index[0]] *
                                node_repr[batch.negative_edge_index[1]], dim=1)

        edgepred_loss = self.criterion(positive_score, torch.ones_like(positive_score)) + \
                        self.criterion(negative_score, torch.zeros_like(negative_score))
        edgepred_acc = (torch.sum(positive_score > 0) +
                        torch.sum(negative_score < 0)).to(torch.float32) / \
                    float(2 * len(positive_score))
        edgepred_acc = edgepred_acc.detach().cpu().item()

        return edgepred_loss, edgepred_acc
