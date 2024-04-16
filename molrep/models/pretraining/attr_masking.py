

import torch
import torch.nn as nn

from molrep.common.registry import registry
from molrep.common.config import Config
from molrep.models.base_model import BaseModel


@registry.register_model("attr_masking")
class AttrMasking(BaseModel):
    """
    Attribution Masking
    """
    MODEL_TYPES_DICT = {
        "pretraining": ("default")
    }

    MODEL_CONFIG_DICT = {
        "default": "configs/models/attr_masking.yaml",
    }

    def __init__(self, model_configs, gnn_model_configs):
        super(AttrMasking, self).__init__()
        self.emb_dim = model_configs['emb_dim']

        self.molecule_gnn_name = model_configs.gnn.model.name
        self.molecule_gnn_model_configs = gnn_model_configs.model
        gnn_model_cls = registry.get_model_class(self.molecule_gnn_name)
        self.molecule_model = gnn_model_cls(
            model_configs=self.molecule_gnn_model_configs,
        )

        self.molecule_atom_masking_model = torch.nn.Linear(self.emb_dim, 119)
        self.criterion = nn.CrossEntropyLoss()

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

        node_repr = self.molecule_model.get_node_representation(data.masked_x, data.edge_index, data.edge_attr)

        attributemask_loss, attributemask_acc = self.do_AttrMasking(batch=data, node_repr=node_repr)

        return {
            'loss': attributemask_loss,
            'acc': attributemask_acc,
        }

    def do_AttrMasking(self, batch, node_repr):
        target = batch.mask_node_label[:, 0]
        try:
            node_pred = self.molecule_atom_masking_model(node_repr[batch.masked_atom_indices])
        except:
            print("The last layer of GNN must equal to the emb_dim")

        attributemask_loss = self.criterion(node_pred.double(), target)
        attributemask_acc = self.compute_accuracy(node_pred, target)
        return attributemask_loss, attributemask_acc

    def compute_accuracy(self, pred, target):
        return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)