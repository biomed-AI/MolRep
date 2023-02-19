
# import torch
# import numpy as np
# from typing import Optional, Text

# from molrep.explainer.base_explainer import BaseExplainer
# from molrep.common.registry import registry

# @registry.register_explainer("random_baseline")
# class RandomBaseline(BaseExplainer):
#     """Random baseline: random node and edge attributions from uniform(0,1)."""

#     def __init__(self, name: Optional[Text] = None):
#         self.name = name or self.__class__.__name__
#         self.sample_size = 1

#     def attribute(self, data, model, **kwargs):
#         """Gets attribtutions."""
#         model.train()
#         output = model(data)
#         if not isinstance(output, tuple):
#             output = (output,)

#         batch_nodes, batch_edges = model.get_node_feats(data)[0], model.get_edge_feats(data)[0]
#         atom_weights = torch.rand(size=(batch_nodes)) if batch_nodes is not None else None
#         bond_weights = torch.rand(size=(batch_edges)) if batch_edges is not None else None

#         model.eval()
#         return atom_weights, bond_weights, output