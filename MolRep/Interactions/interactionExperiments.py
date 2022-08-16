# -*- coding: utf-8 -*-
"""
Created on 2022.06.20

@author: Jiahua Rao

"""


from MolRep.Utils.utils import *
from MolRep.Models.losses import get_loss_func
from MolRep.Utils.config_from_dict import Config
from MolRep.Models.schedulers import build_lr_scheduler
from MolRep.Interactions.interactionNetWrapper import InteractionNetWrapper

class InteractionExperiments:

    def __init__(self, model_configuration, dataset_config, exp_path):
        self.model_config = Config.from_dict(model_configuration) if isinstance(model_configuration, dict) else model_configuration
        self.dataset_config = dataset_config
        self.exp_path = exp_path

        
    def run_valid(self, dataset, logger, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR TRAINING OR EARLY STOPPING!
        :return: (training accuracy, test accuracy)
        """
        model_class = self.model_config.model
        model = model_class(model_configs=self.model_config, dataset_configs=self.dataset_config)
        net = InteractionNetWrapper(model, dataset_configs=self.dataset_config, model_config=self.model_config)

        inputs = model.processing_data(dataset.data, dataset.splits)
        train_loss, valid_results, test_results, elapsed = net.train(inputs=inputs,
                                                                     log_every=10,
                                                                     logger=logger)

        if other is not None and 'model_path' in other.keys():
            save_checkpoint(path=other['model_path'], model=model)

        return train_loss, valid_results, test_results
