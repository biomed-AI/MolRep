# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica, Federico, et al. "A fair comparison of graph neural networks for graph classification." (ICLR 2020). -> https://github.com/diningphil/gnn-comparison
"""

import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch.nn.functional as F

import torch
from MolRep.Models.metrics import *
from MolRep.Utils.utils import *
from MolRep.Explainer.Attribution.utils.utils import *

from collections import defaultdict

def format_time(avg_time):
    avg_time = timedelta(seconds=avg_time)
    total_seconds = int(avg_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(avg_time.microseconds)[:3]}"

class ExplainerXNetWrapper:
    def __init__(self, model, dataset_configs, model_config, loss_function=None):
        self.model = model
        self.loss_fun = loss_function
        self.model_name = model_config.exp_name
        self.dataset_name = dataset_configs["name"]
        self.num_epochs = model_config['num_epochs']

        self.metric_type = dataset_configs["metric_type"]
        self.task_type = dataset_configs["task_type"]
        self.target_cols = dataset_configs["target_columns"]
        self.num_tasks = len(self.target_cols)

        self.device = torch.device(model_config['device'])

        self.print_metric_type = self.metric_type[0] if isinstance(self.metric_type, list) else self.metric_type

    def train(self, train_loader, valid_loader=None, test_loader=None, scaler=None,
              scheduler=None, clipping=None, optimizer=None, early_stopping=None,
              log_every=10, logger=None):
        '''
        Args:
            - train_loader (DataLoader):
            - test_loader (DataLoader):
            - scaler ():
            - scheduler ():
        '''

        early_stopper = early_stopping() if early_stopping is not None else None

        val_loss, val_metric = -1, -1
        test_loss, test_metric = None, None

        time_per_epoch = []

        # Training for Deep learning Methods
        for i in range(1, self.num_epochs+1):
            begin = time.time()
            train_metric, train_loss = self.train_one_epoch(train_loader, optimizer, clipping=clipping)
            end = time.time()
            duration = end - begin
            time_per_epoch.append(end)

            if scheduler is not None:
                scheduler.step()

            if test_loader is not None:
                _, _, test_metric, test_loss = self.test_on_epoch_end(test_loader, scaler)

            if valid_loader is not None:
                _, _, val_metric, val_loss = self.test_on_epoch_end(valid_loader, scaler)
                if early_stopper is not None and early_stopper.stop(i, val_loss, val_acc=val_metric[self.print_metric_type],
                                                                    train_loss=train_loss, train_acc=train_metric[self.print_metric_type]):
                    if logger is not None:
                        logger.log(f'Stopping at epoch {i}, best is {early_stopper.get_best_vl_metrics()}')
                    else:
                        print(f'Stopping at epoch {i}, best is {early_stopper.get_best_vl_metrics()}')
                    break

            if i % log_every == 0 or i == 1:
                logger.log(f'[TRAIN] Epoch: %d, train loss: %.6f train %s: %.6f' % (
                    i, train_loss, self.print_metric_type, train_metric[self.print_metric_type]))
                logger.log(f'[TRAIN] Metric:%s' % (
                  str(train_metric)))
                if valid_loader is not None:
                    logger.log(f'[VALID] Epoch: %d, valid loss: %.6f valid %s: %.6f' % (
                        i, val_loss, self.print_metric_type, val_metric[self.print_metric_type]))
                logger.log(f'[VALID] Metric:%s' % (
                  str(val_metric)))
                if test_loader is not None:
                    logger.log(f'[TEST] Epoch: %d, test loss: %.6f test %s: %.6f' % (
                        i, test_loss, self.print_metric_type, test_metric[self.print_metric_type]))
                logger.log(f'[TEST] Metric:%s' % (
                  str(test_metric)))
                logger.log(f"- Elapsed time: {str(duration)[:4]}s , Time estimation in a fold: {str(duration*self.num_epochs/60)[:4]}min")

        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())

        elapsed = format_time(avg_time_per_epoch)

        if early_stopper is not None:
            train_loss, train_metric, val_loss, val_metric, test_loss, test_metric, best_epoch = early_stopper.get_best_vl_metrics()

        return train_loss, train_metric, val_loss, val_metric, test_loss, test_metric, elapsed

    def test(self, test_loader=None, scaler=None,
             scheduler=None, log_every=10, logger=None):
        
        y_preds, y_labels, test_metric, test_loss = self.test_on_epoch_end(test_loader, scaler)
        logger.log(f'[TEST] test loss: %.6f test %s: %.6f' % ( test_loss, self.metric_type, test_metric))

        return y_preds, y_labels, test_metric

    def explainer(self, test_loader=None, scaler=None, scheduler=None, log_every=0, logger=None):
        model = self.model.to(self.device)
        model.eval()

        smiles_list = test_loader.smiles
        atom_attr_preds, bond_attr_preds = [], []
        y_preds, y_labels = [], []
        loss_all = 0
        for batch_index, data in enumerate(test_loader):

            # Prediction
            if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'GraphNet', 'GAT', 'MorganFP', 'MACCSFP', 'XGraphSAGE', 'XGAT', 'XGIN', 'XMorganFP', 'XMACCSFP']:
                target_batch = data.y
                data = data.to(self.device)

            elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
                mol_batch, features_batch, target_batch, atom_descriptors_batch = data
                data = (mol_batch, features_batch, atom_descriptors_batch)

            else: 
                raise print("Explainer Model Must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MPNN', 'DMPNN', 'CMPNN']")

            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in target_batch])
            labels = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in target_batch])
            class_weights = torch.ones(labels.shape)


            output = model(data)
            if not isinstance(output, tuple):
                output = (output,)

            # Inverse scale if regression
            if scaler is not None:
                output = list(output)
                output[0] = torch.Tensor(scaler.inverse_transform(output[0].detach().cpu().numpy()))
                output = tuple(output)

            if self.task_type == 'Multi-Classification':
                labels = labels.long()
                # loss = torch.cat([self.loss_fun(labels[:, target_index], output[0][:, target_index, :]).unsqueeze(1) for target_index in range(output[0].size(1))], dim=1) * class_weights * mask
                loss = self.loss_fun(labels[:, 0], output[0][:, 0, :]) * class_weights * mask
            else:
                loss = self.loss_fun(labels, *output) * class_weights * mask
            loss = loss.sum() / mask.sum()

            y_preds.extend(output[0].data.cpu().numpy().tolist())
            y_labels.extend(target_batch)
            loss_all += loss.item() * labels.size()[0]

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type)

        return y_preds, y_labels, results

    def train_one_epoch(self, train_loader, optimizer, clipping=None, early_stopping=None):
        model = self.model.to(self.device)
        model.train()

        loss_all = 0
        y_preds, y_labels = [], []
        for _, data in enumerate(train_loader):

            if self.model_name in ['DGCNN', 'GIN', 'ECC', 'DiffPool', 'GraphNet', 'GAT', 'MorganFP', 'MACCSFP', 'XGraphSAGE', 'XGAT', 'XGIN', 'XMorganFP', 'XMACCSFP']:
                target_batch = data.y
                data = data.to(self.device)

            elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
                mol_batch, features_batch, target_batch, atom_descriptors_batch = data
                data = (mol_batch, features_batch, atom_descriptors_batch)

            else:
                raise print("Explainer Model Must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MPNN', 'DMPNN', 'CMPNN']")

            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in target_batch])
            labels = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in target_batch])
            class_weights = torch.ones(labels.shape)

            optimizer.zero_grad()
            output = model(data)

            if not isinstance(output, tuple):
                output = (output,)

            if self.task_type == 'Multi-Classification':
                labels = labels.long()
                # loss = torch.cat([self.loss_fun(labels[:, target_index], output[0][:, target_index, :]).unsqueeze(1) for target_index in range(output[0].size(1))], dim=1) * class_weights * mask
                loss = self.loss_fun(labels[:, 0], output[0][:, 0, :]) * class_weights * mask

            else:
                loss = self.loss_fun(labels, *output) * class_weights * mask


            loss = loss.sum() / mask.sum()
            loss.backward()
            optimizer.step()

            y_preds.extend(output[0].data.cpu().numpy().tolist())
            y_labels.extend(target_batch)

            loss_all += loss.item() * labels.size()[0]

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)

        if self.task_type == 'Classification':
            y_preds = torch.sigmoid(torch.FloatTensor(y_preds))

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type)
        return results, loss_all / len(train_loader.dataset)

    def test_on_epoch_end(self, test_loader, scaler=None):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        y_preds, y_labels = [], []
        for _, data in enumerate(test_loader):
            if self.model_name in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MolecularFingerprint', 'MorganFP']:
                target_batch = data.y
                data = data.to(self.device)

            elif self.model_name in ['MPNN', 'DMPNN', 'CMPNN']:
                mol_batch, features_batch, target_batch, atom_descriptors_batch = data
                data = (mol_batch, features_batch, atom_descriptors_batch)

            else: 
                raise print("Explainer Model Must be in ['DGCNN', 'GIN', 'ECC', 'GraphSAGE', 'DiffPool', 'MPNN', 'DMPNN', 'CMPNN']")

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
                # loss = torch.cat([self.loss_fun(labels[:, target_index], output[0][:, target_index, :]).unsqueeze(1) for target_index in range(output[0].size(1))], dim=1) * class_weights * mask
                loss = self.loss_fun(labels[:, 0], output[0][:, 0, :]) * class_weights * mask
            else:
                loss = self.loss_fun(labels, *output) * class_weights * mask

            loss = loss.sum() / mask.sum()
            loss_all += loss.item() * labels.size()[0]

            y_preds.extend(output[0].data.cpu().numpy().tolist())
            y_labels.extend(target_batch)

        # if self.task_type == 'Multi-Classification':
        #     y_preds = torch.argmax(torch.FloatTensor(y_preds), dim=2)

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type)

        return y_preds, y_labels, results, loss_all / len(test_loader.dataset)


    def evaluate_predictions(self, preds, targets, num_tasks, metric_type, task_type):
        # Filter out empty targets
        # valid_preds and valid_targets have shape (num_tasks, data_size)
        valid_preds = [[] for _ in range(num_tasks)]
        valid_targets = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):
            for j in range(len(preds)):
                if not np.isnan(targets[j][i]):  # Skip those without targets
                    valid_preds[i].append(preds[j][i])
                    valid_targets[i].append(targets[j][i])
            if len(valid_targets[i]) == 0:
                assert print('valid target len is 0')

        if not isinstance(metric_type, list):
            results = {metric_type: []}
        else:
            results = {metric_t: [] for metric_t in metric_type}

        for i in range(num_tasks):
            # # Skip if all targets or preds are identical, otherwise we'll crash during classification
            if task_type == 'Classification':
                nan = False
                if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                    nan = True
                    print('Warning: Found a task with targets all 0s or all 1s')
                if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                    nan = True
                    print('Warning: Found a task with predictions all 0s or all 1s')

                if nan:
                    for metric_t in results.keys():
                        results[metric_t].append(float('nan'))
                    continue

            if len(valid_targets[i]) == 0:
                continue

            metrics_results = get_metric(valid_targets[i], valid_preds[i], metric_type=metric_type)
            for metric_t in results.keys():
                results[metric_t].append(metrics_results[metric_t])

        scores = {key: np.nanmean(results[key]) for key in results}
        return scores
