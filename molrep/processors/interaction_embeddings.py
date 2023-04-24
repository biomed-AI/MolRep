
import os
import numpy as np

import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

import pickle
from itertools import repeat, chain
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from molrep.common.registry import registry


@registry.register_processor("interaction")
class InteractionEmbeddings(InMemoryDataset):
    def __init__(self, cfg, **kwargs):

        self.config = cfg
        
