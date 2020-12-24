# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica, Federico, et al. "A fair comparison of graph neural networks for graph classification." (ICLR 2020). -> https://github.com/diningphil/gnn-comparison
"""

import time

import numpy as np
import pandas as pd

import torch
from MolRep.Models.metrics import *
from MolRep.Utils.utils import *

from collections import defaultdict

class NetWrapper:
    def __init__(self, model, configs, dataset_configs, model_config, loss_function=None):
        self.model = model
        self.loss_fun = loss_function
        self.model_name = configs.model_name
        self.dataset_name = configs.dataset_name
        self.log_file = configs.log_file
        self.num_epochs = model_config['num_epochs']

        self.metric_type = dataset_configs["metric_type"]
        self.task_type = dataset_configs["task_type"]
        self.target_cols = dataset_configs["target_columns"]
        self.num_tasks = len(self.target_cols)

        self.device = torch.device(model_config['device'])
        self.classifer_model = model_config['classifier_model'] if 'classifier_model' in model_config.config_dict.keys() else None


    def train_test_one_fold(self, train_loader, valid_loader=None, test_loader=None, scaler=None,
                            scheduler=None, clipping=None, optimizer=None, early_stopping=None,
                            target_idx=-1, log_every=1, logger=None):
        '''
        Args:
            - train_loader (DataLoader):
            - test_loader (DataLoader):
            - scaler ():
            - scheduler ():
        '''

        # Training for Machine learning Methods
        if self.model_name in ['Mol2Vec', 'N_Gram_Graph'] and self.classifier_model in ['RandomForest', 'XGboost']:
            
            begin = time.time()
            train_x, train_y = get_xy(train_loader, target_idx)
            self.model.fit_model(train_x, train_y)
            y_preds = self.model.predict(train_x)
            metric = get_metric(train_y, y_preds, self.metric_type)

            if valid_loader is not None:
                val_x, val_y = get_xy(valid_loader, target_idx)
                val_y_preds = self.model.predict(val_x)
                val_metric = get_metric(val_y, val_y_preds, self.metric_type)

            end = time.time()
            duration = end - begin

            logger.info(f'[TRAIN] train %s: %.6f' % (self.metric_type, metric))
            if valid_loader is not None:
                logger.info(f'[VALID] valid %s: %.6f' % (self.metric_type, val_metric))
            logger.info(f"- Elapsed time: {str(duration)[:4]}s , Time estimation in a fold: {str(duration*(len(self.target_cols))/60)[:4]}min")


            test_x, test_y = get_xy(test_loader, target_idx)
            test_y_preds = self.model.predict(test_x)
            test_metric = get_metric(test_y, test_y_preds, self.metric_type)
            logger.info(f'[TEST] test %s: %.6f' % (self.metric_type, metric))

            return test_metric, None


        # Training for Deep learning Methods
        for i in range(self.num_epochs):
            begin = time.time()
            metric, loss = self.train_one_epoch(train_loader, optimizer, scheduler, clipping=clipping, logger=logger)

            if valid_loader is not None:
                val_metric, val_loss = self.test_on_epoch_end(valid_loader, scaler, logger=logger)

            end = time.time()
            duration = end - begin

            if i % log_every == 0:
                logger.info(f'[TRAIN] Epoch: %d, train loss: %.6f train %s: %.6f' % (
                    i+1, loss, self.metric_type, metric))
                if valid_loader is not None:
                    logger.info(f'[VALID] Epoch: %d, valid loss: %.6f valid %s: %.6f' % (
                        i+1, val_loss, self.metric_type, val_metric))
                logger.info(f"- Elapsed time: {str(duration)[:4]}s , Time estimation in a fold: {str(duration*self.num_epochs/60)[:4]}min")


        _, metric, loss = self.test_on_epoch_end(test_loader, scaler, logger=logger)
        logger.info(f'[TEST] test loss: %.6f test %s: %.6f' % (
            loss, self.metric_type, metric))
        return metric, loss


    def train_one_epoch(self, train_loader, optimizer, scheduler=None, clipping=None, early_stopping=None, logger=None):
        model = self.model.to(self.device)
        model.train()

        loss_all = 0
        y_preds, y_labels = [], []
        for _, data in enumerate(train_loader):

            if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool']:
                target_batch = data.y
                data = data.to(self.device)

            elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
                mol_batch, features_batch, target_batch, atom_descriptors_batch = data
                data = (mol_batch, features_batch, atom_descriptors_batch)

            elif self.model_name == 'MAT':
                target_batch = data[-1]
                data = [features.to(self.device) for features in data]

            elif self.model_name in ['BiLSTM', 'SALSTM', 'Transformer']:
                target_batch = data[-1]
                data = (data[0].to(self.device), data[1].to(self.device))

            elif self.model_name == 'VAE':
                target_batch = data[-1]
                data = tuple(d.to(self.device) for d in data[0])

            else:
                raise self.logger.error(f"Model Name must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', \
                                                'MPNN', 'DMPNN', 'CMPNN', 'MAT', 'BiLSTM', 'SALSTM', 'Transformer', 'VAE', 'Mol2Vec', 'N-Gram-Graph']")

            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in target_batch])
            labels = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in target_batch])
            class_weights = torch.ones(labels.shape)

            optimizer.zero_grad()
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)

            if self.task_type == 'Multi-Classification':
                labels = labels.long()
                loss = torch.cat([self.loss_fun(labels[:, target_index], output[0][:, target_index, :]).unsqueeze(1) for target_index in range(output[0].size(1))], dim=1) * class_weights * mask
            else:
                loss = self.loss_fun(labels, *output) * class_weights * mask
            loss = loss.sum() / mask.sum()

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            y_preds.extend(output[0].data.cpu().numpy().tolist())
            y_labels.extend(target_batch)

            loss_all += loss.item() * labels.size()[0]

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type, logger=logger)

        return results, loss_all / len(train_loader.dataset)

    def test_on_epoch_end(self, test_loader, scaler=None, logger=None):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        y_preds, y_labels = [], []
        for _, data in enumerate(test_loader):
            if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool']:
                target_batch = data.y
                data = data.to(self.device)

            elif self.model_name == 'MAT':
                target_batch = data[-1]
                data = [features.to(self.device) for features in data]

            elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
                mol_batch, features_batch, target_batch, atom_descriptors_batch = data
                data = (mol_batch, features_batch, atom_descriptors_batch)

            elif self.model_name in ['BiLSTM', 'SALSTM', 'Transformer']:
                target_batch = data[-1]
                data = (data[0].to(self.device), data[1].to(self.device))

            elif self.model_name == 'VAE':
                target_batch = data[-1]
                data = data[0].to(self.device)

            else:
                raise self.logger.error(f"Model Name must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', \
                                                'MPNN', 'DMPNN', 'CMPNN', 'MAT', 'BiLSTM', 'SALSTM', 'Transformer', 'VAE', 'Mol2Vec', 'N-Gram-Graph']")


            output = model(data)
            if not isinstance(output, tuple):
                output = (output,)

            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in target_batch])
            labels = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in target_batch])
            class_weights = torch.ones(labels.shape)

            # Inverse scale if regression
            if scaler is not None:
                output = list(output)
                output[0] = torch.Tensor(scaler.inverse_transform(output[0].detach().cpu().numpy()))
                output = tuple(output)

            if self.task_type == 'Multi-Classification':
                labels = labels.long()
                loss = torch.cat([self.loss_fun(labels[:, target_index], output[0][:, target_index, :]).unsqueeze(1) for target_index in range(output[0].size(1))], dim=1) * class_weights * mask
            else:
                loss = self.loss_fun(labels, *output) * class_weights * mask
            loss = loss.sum() / mask.sum()

            y_preds.extend(output[0].data.cpu().numpy().tolist())
            y_labels.extend(target_batch)
            loss_all += loss.item() * labels.size()[0]

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type, logger=logger)

        return y_preds, results, loss_all / len(test_loader.dataset)


    def evaluate_predictions(self, preds, targets, num_tasks, metric_type, task_type, logger=None):

        # Filter out empty targets
        # valid_preds and valid_targets have shape (num_tasks, data_size)
        valid_preds = [[] for _ in range(num_tasks)]
        valid_targets = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):
            for j in range(len(preds)):
                if not np.isnan(targets[j][i]):  # Skip those without targets
                    valid_preds[i].append(preds[j][i])
                    valid_targets[i].append(targets[j][i])

        results = []
        for i in range(num_tasks):
            # # Skip if all targets or preds are identical, otherwise we'll crash during classification
            if task_type == 'Classification':
                nan = False
                if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                    nan = True
                    logger.info('Warning: Found a task with targets all 0s or all 1s')
                if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                    nan = True
                    logger.info('Warning: Found a task with predictions all 0s or all 1s')

                if nan:
                    results.append(float('nan'))
                    continue

            if len(valid_targets[i]) == 0:
                continue

            results.append(get_metric(valid_targets[i], valid_preds[i], metric_type=metric_type))

        scores = np.nanmean(results)
        return scores

    def get_xy(loader, target_idx):

        x_all, y_all = loader
        target_y = y_all[:, target_idx]
    
        nan_rows = np.where(np.isnan(target_y))[0]
        x_valid = [x_all[i] for i in range(len(x_all)) if i not in nan_rows]
        y_valid = [target_y[i] for i in range(len(target_y)) if i not in nan_rows]
        return x_valid, y_valid