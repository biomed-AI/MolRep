
import os
import random

from MolRep.Models.losses import get_loss_func
from MolRep.Models.netWrapper import NetWrapper
from MolRep.Models.schedulers import build_lr_scheduler

from MolRep.Utils.config_from_dict import Config, DatasetConfig
from MolRep.Utils.utils import *

class Experiment:
    """
    Experiment provides a layer of abstraction to avoid that all models implement the same interface
    """

    def __init__(self, model_configuration, dataset_config, exp_path):
        self.model_config = Config.from_dict(model_configuration)
        self.dataset_config = dataset_config
        self.exp_path = exp_path

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

    def run_valid(self, get_train_val, logger, other=None):
        """
        This function returns the training and validation accuracy. DO WHATEVER YOU WANT WITH VL SET,
        BECAUSE YOU WILL MAKE PERFORMANCE ASSESSMENT ON A TEST SET
        :return: (training accuracy, validation accuracy)
        """
        raise NotImplementedError('You must implement this function!')

    def run_test(self, get_train_val, get_test, logger, other=None):
        """
        This function returns the training and test accuracy
        :return: (training accuracy, test accuracy)
        """
        raise NotImplementedError('You must implement this function!')



class EndToEndExperiment(Experiment):

    def __init__(self, model_configuration, dataset_config, exp_path):
        super(EndToEndExperiment, self).__init__(model_configuration, dataset_config, exp_path)


    def run_valid(self, dataset_getter, logger, other=None):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        # print(self.model_config, dataset_getter.outer_k, dataset_getter.inner_k)

        dataset = dataset_getter.get_dataset
        model_class = self.model_config.model
        optim_class = self.model_config.optimizer
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        loss_fn = get_loss_func(self.dataset_config['task_type'], self.model_config.exp_name)
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True


        train_loader, val_loader, scaler = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)

        model = model_class(dim_features=dataset.dim_features, dim_target=dataset.dim_target, model_configs=self.model_config, dataset_configs=self.dataset_config)
        net = NetWrapper(model, dataset_configs=self.dataset_config, model_config=self.model_config,
                         loss_function=loss_fn)

        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])
        scheduler = build_lr_scheduler(optimizer, model_configs=self.model_config, num_samples=dataset.num_samples)


        train_loss, train_metric, val_loss, val_metric, _, _, _ = net.train(train_loader=train_loader,
                                                                            optimizer=optimizer, scheduler=scheduler,
                                                                            clipping=clipping, scaler=scaler,
                                                                            valid_loader=val_loader,
                                                                            early_stopping=stopper_class,
                                                                            logger=logger)

        if other is not None and 'model_path' in other.keys():
            save_checkpoint(path=other['model_path'], model=model, scaler=scaler)

        return train_metric, val_metric


    def run_test(self, dataset_getter, logger, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR TRAINING OR EARLY STOPPING!
        :return: (training accuracy, test accuracy)
        """
        dataset = dataset_getter.get_dataset
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        model_class = self.model_config.model
        optim_class = self.model_config.optimizer
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        loss_fn = get_loss_func(self.dataset_config['task_type'], self.model_config.exp_name)
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True


        train_loader, val_loader, scaler = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)
        test_loader = dataset_getter.get_test(dataset, self.model_config['batch_size'], shuffle=shuffle)

        
        model = model_class(dim_features=dataset.dim_features, dim_target=dataset.dim_target, model_configs=self.model_config, dataset_configs=self.dataset_config)
        net = NetWrapper(model, dataset_configs=self.dataset_config, model_config=self.model_config,
                         loss_function=loss_fn)
        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])
        scheduler = build_lr_scheduler(optimizer, model_configs=self.model_config, num_samples=dataset.num_samples)

        train_loss, train_metric, val_loss, valid_metric, test_loss, test_metric, _ = \
            net.train(train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader, 
                      optimizer=optimizer, scheduler=scheduler, clipping=clipping,
                      early_stopping=stopper_class, scaler=scaler,
                      logger=logger)

        if other is not None and 'model_path' in other.keys():
            save_checkpoint(path=other['model_path'], model=model, scaler=scaler)

        return train_metric, test_metric


    def run_independent_test(self, dataset_getter, logger, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR TRAINING OR EARLY STOPPING!
        :return: (training accuracy, test accuracy)
        """
        dataset = dataset_getter.get_dataset
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        model_class = self.model_config.model
        optim_class = self.model_config.optimizer
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        loss_fn = get_loss_func(self.dataset_config['task_type'], self.model_config.exp_name)
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True


        test_loader = dataset_getter.get_test(dataset, self.model_config['batch_size'], shuffle=shuffle)

        model = model_class(dim_features=dataset.dim_features, dim_target=dataset.dim_target, model_configs=self.model_config, dataset_configs=self.dataset_config)
        net = NetWrapper(model, dataset_configs=self.dataset_config, model_config=self.model_config,
                         loss_function=loss_fn)
        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])
        scheduler = build_lr_scheduler(optimizer, model_configs=self.model_config, num_samples=dataset.num_samples)


        if other is not None and 'model_path' in other.keys():
            model = load_checkpoint(path=other['model_path'], model=model)
            scaler, features_scaler = load_scalers(path=other['model_path'])

        y_preds, y_labels, test_metric = net.test(test_loader=test_loader, scaler=scaler, logger=logger)

        return y_preds, y_labels, test_metric