

import os
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from molrep.common.registry import registry
from molrep.common.config import Config
from molrep.common.dist_utils import download_cached_file
from molrep.common.utils import is_url

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
    def default_config_path(cls, model_type='default'):
        return os.path.join(registry.get_path("library_root"), cls.MODEL_CONFIG_DICT[model_type])

    @classmethod
    def best_config_path(cls, property_name='ogbg-molbbbp'):
        return os.path.join(registry.get_path("library_root"), cls.MODEL_CONFIG_DICT['best'][property_name])

    @classmethod
    def from_config(cls, cfg=None):
        raise NotImplementedError

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        print("Missing keys {}".format(msg.missing_keys))
        print("load checkpoint from %s" % url_or_filename)
        return msg


    @classmethod
    def from_pretrained(cls, property_name, checkpoint=None):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        if 'best' in cls.MODEL_CONFIG_DICT.keys() and property_name in cls.MODEL_CONFIG_DICT['best']:
            cfg_path = cls.best_config_path(property_name)
        else:
            cfg_path = cls.default_config_path()

        cfg = Config.build_best_model_configs(cfg_path = cfg_path)
        model = cls.from_config(cfg)

        if checkpoint is None:
            checkpoint = os.path.join(registry.get_path("repo_root"), cfg.model_cfg.trained)

        if checkpoint != "":
            model.load_checkpoint(checkpoint)
        return model, cfg

    def featurize(self, data):
        pass

    @torch.no_grad()
    def predict(self, loader, task='classification', **kwargs):
        predictions = []
        for idx, batch in enumerate(loader):
            for k, v in batch.items():
                if type(v) == torch.Tensor or issubclass(type(v), Batch):
                    batch[k] = v.to(self.device, non_blocking=True)
            output = self.forward(batch).logits
            if task == 'classification':
                output = output.sigmoid()
            if task == 'multiclass-classification':
                output = output.reshape((output.size(0), -1, kwargs['num_classes']))
                multiclass_softmax = nn.Softmax(dim=2)
                output = multiclass_softmax(output)
            predictions.append(output.cpu().data.numpy())
        predictions = np.concatenate(predictions, axis=0)
        return predictions

    def get_gradients(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_node_feats(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_edge_feats(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_gap_activations(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_prediction_weights(self):
        raise NotImplementedError("The model does not implement the Explainer function.")

    def get_intermediate_activations_gradients(self, data):
        raise NotImplementedError("The model does not implement the Explainer function.")



from molrep.processors.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i].long())
        return x_embedding


class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i].long())
        return bond_embedding


import torch

import numpy as np
from dataclasses import fields

from dataclasses import dataclass
from typing import Optional

from collections import OrderedDict
from typing import Any, ContextManager, List, Tuple


def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray` or `np.ndarray`.
    """
    if isinstance(x, torch.Tensor):
        return True
    return isinstance(x, np.ndarray)

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.
    <Tip warning={true}>
    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.
    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class ModelOutputs(ModelOutput):
    
    node_features: torch.FloatTensor = None
    edge_features: torch.FloatTensor = None
    logits: torch.FloatTensor = None


@dataclass
class ModelOutputsWithLoss(ModelOutput):

    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
