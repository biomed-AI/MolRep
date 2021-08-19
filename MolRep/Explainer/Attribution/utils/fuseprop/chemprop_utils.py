from typing import List

import torch
import torch.nn as nn
from tqdm import trange


from MolRep.Models.scalers import StandardScaler
from MolRep.Experiments.Graph_Data.MPNN_data import *


def get_data_from_smiles(smiles: List[str], logger = None) -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.

    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    """
    data = MoleculeDataset([MoleculeDatapoint(smiles=smile) for smile in smiles])
    return data

def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch
        batch_data = (batch, features_batch, None)

        with torch.no_grad():
            batch_preds = model(batch_data)

        batch_preds = batch_preds.data.cpu().numpy()
        
        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
