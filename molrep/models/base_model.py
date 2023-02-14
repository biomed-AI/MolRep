

import os
import torch.nn as nn

from molrep.common.registry import registry

class BaseModel(nn.Module):

    MODEL_CONFIG_DICT = {
        "base_model": "",
    }

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    @classmethod
    def default_config_path(cls, model_type):
        return os.path.join(registry.get_path("library_root"), cls.MODEL_CONFIG_DICT[model_type])

    @classmethod
    def from_config(cls, cfg=None):
        raise NotImplementedError

    def featurize(self, data):
        pass

    def get_gradients(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_batch_nums(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_gap_activations(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_prediction_weights(self):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_intermediate_activations_gradients(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")



