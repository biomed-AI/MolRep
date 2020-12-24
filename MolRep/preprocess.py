# -*- coding: utf-8 -*-
'''
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie
'''


import numpy as np
import pandas as pd

from MolRep.Featurization.Graph_embeddings import GraphEmbeddings
from MolRep.Featurization.MPNN_embeddings import MPNNEmbeddings

from MolRep.Featurization.MAT_embeddings import MATEmbeddings
from MolRep.Featurization.VAE_embeddings import VAEEmbeddings
from MolRep.Featurization.Sequence_embeddings import SequenceEmbeddings

from MolRep.Featurization.Mol2Vec_embeddings import Mol2VecEmbeddings
from MolRep.Featurization.NGramGraph_embeddings import NGramGraphEmbeddings

def prepare_features(configs, dataset_configs, logger):
    '''
        Args:
            - configs (Namespace): Namespace of basic configuration.
            - dataset_configs (dict): Namespace of dataset configuration.
            - logger (logging): logging.
    '''

    dataset_path = configs.dataset_path
    features_dir = configs.features_dir

    if configs.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool']:
        preparer = GraphEmbeddings(dataset_path=dataset_path,
                                  features_dir=features_dir,
                                  configs=configs,
                                  dataset_configs=dataset_configs,
                                  logger=logger)
        preparer.process()

    elif configs.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
        preparer = MPNNEmbeddings(dataset_path=dataset_path,
                                  features_dir=features_dir,
                                  configs=configs,
                                  dataset_configs=dataset_configs,
                                  logger=logger)
        preparer.process()

    elif configs.model_name == 'MAT':
        preparer = MATEmbeddings(dataset_path=dataset_path,
                                 features_dir=features_dir,
                                 configs=configs,
                                 dataset_configs=dataset_configs,
                                 logger=logger)
        preparer.process()

    elif configs.model_name in ['BiLSTM', 'SALSTM', 'Transformer']:
        preparer = SequenceEmbeddings(dataset_path=dataset_path,
                                      features_dir=features_dir,
                                      configs=configs,
                                      dataset_configs=dataset_configs,
                                      logger=logger)
        preparer.process()

    elif configs.model_name == 'VAE':
        preparer = VAEEmbeddings(dataset_path=dataset_path,
                                 features_dir=features_dir,
                                 configs=configs,
                                 dataset_configs=dataset_configs,
                                 logger=logger)
        preparer.process()

    elif configs.model_name == 'Mol2Vec':
        preparer = Mol2VecEmbeddings(dataset_path=dataset_path,
                                     mol2vec_model_path=configs.mol2vec_model_path,
                                     features_dir=features_dir,
                                     configs=configs,
                                     dataset_configs=dataset_configs,
                                     logger=logger)
        preparer.process()

    elif configs.model_name == 'N_Gram_Graph':
        preparer = NGramGraphEmbeddings(dataset_path=dataset_path,
                                        features_dir=features_dir,
                                        configs=configs,
                                        dataset_configs=dataset_configs,
                                        logger=logger)
        preparer.process()