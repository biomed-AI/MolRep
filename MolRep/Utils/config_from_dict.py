# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Shuangjia Zheng, Hui Yang, Jiancong Xie
'''

from copy import deepcopy
from torch.optim import Adam, SGD
from MolRep.Models.early_stoppers import GLStopper, Patience

from MolRep.Models.sequence_based.MAT import MAT
from MolRep.Models.sequence_based.CoMPT import CoMPT
from MolRep.Models.sequence_based.BiLSTM import BiLSTM
from MolRep.Models.sequence_based.SALSTM import SALSTM
from MolRep.Models.sequence_based.Transformer import Transformer

from MolRep.Models.graph_based.GIN import GIN
from MolRep.Models.graph_based.ECC import ECC
from MolRep.Models.graph_based.DGCNN import DGCNN
from MolRep.Models.graph_based.DiffPool import DiffPool
from MolRep.Models.graph_based.GraphSAGE import GraphSAGE
from MolRep.Models.graph_based.MolecularFingerprint import MolecularFingerprint

from MolRep.Models.graph_based.MPNN import MPNN
from MolRep.Models.graph_based.CMPNN import CMPNN
from MolRep.Models.graph_based.DMPNN import DMPNN

from MolRep.Models.unsupervised_based.VAE import VAE
from MolRep.Models.unsupervised_based.RandomForest import RandomForest
from MolRep.Models.unsupervised_based.XGboost import XGboost

from MolRep.Utils.utils import read_config_file

class Config:
    """
    Specifies the configuration for a single model.
    """

    models = {
        # graph based methods
        'GIN': GIN,
        'ECC': ECC,
        'DGCNN': DGCNN,
        'DiffPool': DiffPool,
        'GraphSAGE': GraphSAGE,
        'MolecularFingerprint': MolecularFingerprint,

        'MPNN': MPNN,
        'CMPNN': CMPNN,
        'DMPNN': DMPNN,

        # sequences based methods
        'MAT': MAT,
        'CoMPT': CoMPT,
        'BiLSTM': BiLSTM,
        'SALSTM': SALSTM,
        'Transformer': Transformer,

        # unspervised based methods
        'VAE': VAE,
        'RandomForest': RandomForest,
        'XGboost': XGboost,
    }

    optimizers = {
        'Adam': Adam,
        'SGD': SGD
    }

    early_stoppers = {
        'GLStopper': GLStopper,
        'Patience': Patience
    }

    def __init__(self, **attrs):

        # print(attrs)
        self.config = dict(attrs)

        for attrname, value in attrs.items():
            if attrname in ['model', 'optimizer', 'early_stopper']:
                if attrname == 'model':
                    setattr(self, 'model_name', value)
                fn = getattr(self, f'parse_{attrname}')
                setattr(self, attrname, fn(value))
            else:
                setattr(self, attrname, value)

    def __getitem__(self, name):
        # print("attr", name)
        return getattr(self, name)

    def __contains__(self, attrname):
        return attrname in self.__dict__

    def __repr__(self):
        name = self.__class__.__name__
        return f'<{name}: {str(self.__dict__)}>'


    @property
    def exp_name(self):
        return f'{self.model_name}'

    @property
    def config_dict(self):
        return self.config

    @staticmethod
    def parse_model(model_s):
        assert model_s in Config.models, f'Could not find {model_s} in dictionary!'
        return Config.models[model_s]

    @staticmethod
    def parse_optimizer(optim_s):
        assert optim_s in Config.optimizers, f'Could not find {optim_s} in dictionary!'
        return Config.optimizers[optim_s]

    @staticmethod
    def parse_early_stopper(stopper_dict):
        if stopper_dict is None:
            return None

        stopper_s = stopper_dict['class']
        args = stopper_dict['args']

        assert stopper_s in Config.early_stoppers, f'Could not find {stopper_s} in early stoppers dictionary'

        return lambda: Config.early_stoppers[stopper_s](**args)

    @staticmethod
    def parse_gradient_clipping(clip_dict):
        if clip_dict is None:
            return None
        args = clip_dict['args']
        clipping = None if not args['use'] else args['value']
        return clipping

    @classmethod
    def from_dict(cls, dict_obj):
        return Config(**dict_obj)



class Grid:
    """
    Specifies the configuration for multiple models.
    """

    def __init__(self, path_or_dict):
        self.configs_dict = read_config_file(path_or_dict)
        self.num_configs = 0  # must be computed by _create_grid
        self._configs = self._create_grid()

    def __getitem__(self, index):
        return self._configs[index]

    def __len__(self):
        return self.num_configs

    def __iter__(self):
        assert self.num_configs > 0, 'No configurations available'
        return iter(self._configs)

    def _grid_generator(self, cfgs_dict):
        keys = cfgs_dict.keys()
        result = {}

        if cfgs_dict == {}:
            yield {}
        else:
            configs_copy = deepcopy(cfgs_dict)  # create a copy to remove keys

            # get the "first" key
            param = list(keys)[0]
            del configs_copy[param]

            first_key_values = cfgs_dict[param]
            for value in first_key_values:
                result[param] = value

                for nested_config in self._grid_generator(configs_copy):
                    result.update(nested_config)
                    yield deepcopy(result)

    def _create_grid(self):
        '''
        Takes a dictionary of key:list pairs and computes all possible permutations.
        :param configs_dict:
        :return: A dictionary generator
        '''
        config_list = [cfg for cfg in self._grid_generator(self.configs_dict)]
        self.num_configs = len(config_list)
        return config_list



class DatasetConfig:
    """
    Specifies the configuration for a dataset.
    """
    Data = {
        # Quantum Mechanics
        'QM7b': {
        },
        'QM8': {
            'name': 'QM8',
            'path': 'MolRep/Datasets/QM8/qm8.sdf',
            'smiles_column': 'smiles',
            'target_path': 'qm8.sdf.csv',
            'target_columns': ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-PBE0.1', 'E2-PBE0.1', 'f1-PBE0.1', 'f2-PBE0.1', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'],
            'task_type': 'Regression',
            'metric_type': 'mae',
            'split_type': 'random'
        },
        'QM9': {
            'name': 'QM9',
            'path': 'MolRep/Datasets/QM9/gdb9.sdf',
            'smiles_column': 'smiles',
            'target_path': 'gdb9.sdf.csv',
            'target_columns': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv'],
            'task_type': 'Regression',
            'metric_type': 'mae',
            'split_type': 'random'
        },

        # Physical Chemistry
        'ESOL': {
            'name': 'ESOL',
            'path': 'MolRep/Datasets/ESOL/delaney-processed.csv',
            'smiles_column': 'smiles',
            'target_columns': ['measured log solubility in mols per litre'],
            'task_type': 'Regression',
            'metric_type': 'rmse',
            'split_type': 'random'
        },
        'FreeSolv': {
            'name': 'FreeSolv',
            'path': 'MolRep/Datasets/FreeSolv/SAMPL.csv',
            'smiles_column': 'smiles',
            'target_columns': ['expt'],
            'task_type': 'Regression',
            'metric_type': 'rmse',
            'split_type': 'random'
        },
        'Lipophilicity': {
            'name': 'Lipophilicity',
            'path': 'MolRep/Datasets/Lipophilicity/Lipophilicity.csv',
            'smiles_column': 'smiles',
            'target_columns': ['exp'],
            'task_type': 'Regression',
            'metric_type': 'rmse',
            'split_type': 'random'
        },

        # Biophysics
        'PCBA': {
            'name': 'PCBA',
            'path': 'MolRep/Datasets/PCBA/pcba.csv',
            'smiles_column': 'smiles',
            'target_columns': ['PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457', 'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469', 'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688', 'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242', 'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546', 'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676', 'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294', 'PCBA-485297', 'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349', 'PCBA-485353', 'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 'PCBA-493208', 'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339', 'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842', 'PCBA-504845', 'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317', 'PCBA-588342', 'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590', 'PCBA-588591', 'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233', 'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171', 'PCBA-624173', 'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288', 'PCBA-624291', 'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635', 'PCBA-651644', 'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104', 'PCBA-652105', 'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979', 'PCBA-720504', 'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553', 'PCBA-720579', 'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709', 'PCBA-720711', 'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881', 'PCBA-883', 'PCBA-884', 'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899', 'PCBA-902', 'PCBA-903', 'PCBA-904', 'PCBA-912', 'PCBA-914', 'PCBA-915', 'PCBA-924', 'PCBA-925', 'PCBA-926', 'PCBA-927', 'PCBA-938', 'PCBA-995'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'random'
        },
        'MUV': {
            'name': 'MUV',
            'path': 'MolRep/Datasets/MUV/muv.csv',
            'smiles_column': 'smiles',
            'target_columns': ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692','MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846','MUV-852','MUV-858','MUV-859'],
            'task_type': 'Classification',
            'metric_type': 'prc',
            'split_type': 'random'
        },
        'HIV': {
            'name': 'HIV',
            'path': 'MolRep/Datasets/HIV/HIV.csv',
            'smiles_column': 'smiles',
            'target_columns': ['HIV_active'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'scaffold'
        },
        'PDBbind': {

        },
        'BACE': {
            'name': 'BACE',
            'path': 'MolRep/Datasets/BACE/bace.csv',
            'smiles_column': 'mol',
            'target_columns': ['Class'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'scaffold'
        },

        # Physiology
        'BBBP': {
            'name': 'BBBP',
            'path': 'MolRep/Datasets/BBBP/BBBP.csv',
            'smiles_column': 'smiles',
            'target_columns': ['p_np'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'scaffold'
        },
        'Microdrug': {
            # 'path': 'BBBP/BBBP.csv',
            # 'smiles_column': 'smiles',
            # 'target_columns': ['p_np'],
            'name': 'Microdrug',
            'path': 'MolRep/Datasets/BBBP/microdrug.csv',
            'smiles_column': 'smi',
            'target_columns': ['label'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'random'
        },
        'Tox21': {
            'name': 'Tox21',
            'path': 'MolRep/Datasets/Tox21/tox21.csv',
            'smiles_column': 'smiles',
            'target_columns': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'random'
        },
        'SIDER':{
            'name': 'SIDER',
            'path': 'MolRep/Datasets/SIDER/sider.csv',
            'smiles_column': 'smiles',
            'target_columns': ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders', 'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 'General disorders and administration site conditions', 'Endocrine disorders', 'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders', 'Injury, poisoning and procedural complications'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'random'
        },
        'ClinTox':{
            'name': 'ClonTox',
            'path': 'MolRep/Datasets/ClinTox/clintox.csv',
            'smiles_column': 'smiles',
            'target_columns': ['FDA_APPROVED', 'CT_TOX'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'random'
        },

        'Toy':{
            'name': 'Toy',
            'path': 'MolRep/Datasets/Toy/toy_label_1_mw350.csv',
            'smiles_column': 'SMILES',
            'target_columns': ['label_full'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'random'
        },

        'OS_cell':{
            'name': 'OS_cell',
            'path': 'MolRep/Datasets/OS_cell/os_cell.csv',
            'smiles_column':'Smiles',
            'target_columns': ['pChEMBL Value'],
            'task_type': 'Regression',
            'metric_type': 'rmse',
            'split_type': 'specific',
            'additional_info': {'cell_type_column': 'Assay Cell Type'}
        },

        'OS_cell_clf':{
            'name': 'OS_cell_clf',
            'path': 'MolRep/Datasets/OS_cell/os_cell.csv',
            'smiles_column':'Smiles',
            'target_columns': ['pChEMBL'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'random'
        },

        'OS_ind':{
            'name': 'OS_ind',
            'path': 'MolRep/Datasets/OS_ind/ind_szy.csv',
            'smiles_column':'Smi',
            'target_columns': ['Label'],
            'task_type': 'Regression',
            'metric_type': 'rmse',
            'split_type': 'random',
        },

        'OS_ind_clf':{
            'name': 'OS_ind_clf',
            'path': 'MolRep/Datasets/OS_ind/ind_szy.csv',
            'smiles_column':'Smi',
            'target_columns': ['Label'],
            'task_type': 'Classification',
            'metric_type': 'auc',
            'split_type': 'random',
            'additional_info': {'cell_type_column': 'Assay Cell Type'}
        },
    }

    def __init__(self, dataset_name, data_dict=None):
        try:
            self.dataset_name = dataset_name
            self.dataset_config = dict(DatasetConfig.Data[dataset_name])
        except ValueError as e:
            # print("Dataset name should be in ['QM7b', 'QM8', 'QM9', 'ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', \
            #                         'HIV', 'PDBbind', 'BACE', 'BBBP', 'Tox21', 'SIDER', 'ClinTox']\n")
            self.dataset_name = data_dict['name']
            self.dataset_config = data_dict

    def __getitem__(self, name):
        # print("attr", name)
        return self.dataset_config[name]

    def __contains__(self, attrname):
        return attrname in self.dataset_config.keys()

    @property
    def exp_name(self):
        return f'{self.dataset_name}'
