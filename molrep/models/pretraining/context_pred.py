

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from molrep.common.registry import registry
from molrep.common.config import Config
from molrep.models.base_model import BaseModel


@registry.register_model("context_pred")
class ContextPred(BaseModel):
    """
    Attribution Masking
    """

    MODEL_CONFIG_DICT = {
        "default": "configs/models/context_pred.yaml",
    }

    def __init__(self, model_configs, gnn_model_configs):
        self.model_configs = model_configs

        self.molecule_gnn_type = model_configs.gnn['gnn_type']
        self.molecule_gnn_model_configs = gnn_model_configs
        gnn_model_cls = registry.get_model_class(self.molecule_gnn_type)

        l1 = self.molecule_gnn_model_configs.num_layers - 1
        l2 = l1 + model_configs.csize
        
        self.molecule_substruct_model = gnn_model_cls.from_config(
            model_configs=self.molecule_gnn_model_configs,
        )
        self.molecule_gnn_model_configs.num_layers = int(l2 - l1)
        self.molecule_context_model = gnn_model_cls.from_config(
            model_configs=self.molecule_gnn_model_configs,
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.molecule_readout_func = global_mean_pool

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

        contextpred_loss, contextpred_acc = self.do_ContextPred(data)

        return {
            'contextpred_loss': contextpred_loss, 'contextpred_acc': contextpred_acc
        }


    def do_ContextPred(self, batch):

        # creating substructure representation
        substruct_repr = self.molecule_substruct_model(
            batch.x_substruct, batch.edge_index_substruct,
            batch.edge_attr_substruct)[batch.center_substruct_idx]

        # creating context representations
        overlapped_node_repr = self.molecule_context_model(
            batch.x_context, batch.edge_index_context,
            batch.edge_attr_context)[batch.overlap_context_substruct_idx]

        # positive context representation
        # readout -> global_mean_pool by default
        context_repr = self.molecule_readout_func(overlapped_node_repr,
                                                  batch.batch_overlapped_context)

        # negative contexts are obtained by shifting
        # the indices of context embeddings
        neg_context_repr = torch.cat(
            [context_repr[cycle_index(len(context_repr), i + 1)]
            for i in range(self.model_configs.contextpred_neg_samples)], dim=0)

        num_neg = self.model_configs.contextpred_neg_samples
        pred_pos = torch.sum(substruct_repr * context_repr, dim=1)
        pred_neg = torch.sum(substruct_repr.repeat((num_neg, 1)) * neg_context_repr, dim=1)

        loss_pos = self.criterion(pred_pos.double(),
                            torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = self.criterion(pred_neg.double(),
                            torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        contextpred_loss = loss_pos + num_neg * loss_neg

        num_pred = len(pred_pos) + len(pred_neg)
        contextpred_acc = (torch.sum(pred_pos > 0).float() +
                        torch.sum(pred_neg < 0).float()) / num_pred
        contextpred_acc = contextpred_acc.detach().cpu().item()

        return contextpred_loss, contextpred_acc


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr
