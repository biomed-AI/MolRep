


import torch
from molrep.common.registry import registry
from molrep.dataset.datasets.base_dataset import MoleculeDataset
from molrep.dataset.datasets.pretrain_batch import BatchAE, BatchMasking, BatchSubstructContext, BatchSubstructContext3D


@registry.register_dataset("pretraining")
class PretrainingDataset(MoleculeDataset):

    model_dataloader_mapping = {
        'attr_masking': BatchMasking,
        'edge_pred': BatchAE,
        'context_pred': BatchSubstructContext,
    }

    def __init__(self, data, transform):
        super(PretrainingDataset, self).__init__(data)
        self._data = data
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).
        return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, index):
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.
        Args:
            item: An index (int) or a slice object.
        return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        data = self._data[index]
        data = data if self.transform is None else self.transform(data)
        return data

    @classmethod
    def construct_dataset(cls, features_path, **kwargs):
        features = torch.load(features_path)
        transform = kwargs['transform']
        return cls(features, transform)

    def collate_fn(self, batch, **kwargs):
        pretrain_model_name = kwargs['pretrain_model_name']
        batch_data = self.model_dataloader_mapping[pretrain_model_name].from_data_list(batch)
        return {
            "pygdata": batch_data,
        }

