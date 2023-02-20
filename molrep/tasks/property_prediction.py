#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy.
 All rights reserved.

 @author: Jiahua Rao, Jiancong Xie, Junjie Xie
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


import numpy as np

import torch
from torch_geometric.data import Batch

from molrep.common.registry import registry
from molrep.models.metrics import get_metric
from molrep.tasks.base_task import BaseTask


@registry.register_task("property_prediction")
class PropertyTask(BaseTask):
    def __init__(self, task_type, num_tasks, metric_type, multiclass_num_classes = 1):
        super().__init__()

        self.task_type = task_type
        self.num_tasks = num_tasks
        self.metric_type = [str(metric_type)] if isinstance(metric_type, str) else list(metric_type)

        self.classification = self.task_type == 'classification'
        self.multiclass = self.task_type == 'multiclass-classification'
        self.regression = self.task_type == 'regression'
        self.multiclass_num_classes = multiclass_num_classes
        assert not (self.classification and self.regression and self.multiclass)

    @classmethod
    def setup_task(cls, cfg):
        task_type = cfg.datasets_cfg.task_type
        num_tasks = cfg.datasets_cfg.num_tasks
        metric_type = cfg.datasets_cfg.metric_type
        multiclass_num_classes = cfg.datasets_cfg.get("multiclass_num_classes", 1)

        return cls(
            task_type = task_type,
            num_tasks = num_tasks,
            metric_type = metric_type,
            multiclass_num_classes = multiclass_num_classes,
        )

    def build_datasets(self, cfg):
        return super().build_datasets(cfg)

    def train_epoch(
            self,
            epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            loss_func,
            device="cpu",
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.
        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """

        model = model.to(device)
        model.train()

        loss_all = 0
        y_preds, y_labels = [], []
        for i, batch_data in enumerate(data_loader):

            target_batch = batch_data["targets"]
            for k, v in batch_data.items():
                if type(v) == torch.Tensor or issubclass(type(v), Batch):
                    batch_data[k] = v.to(device, non_blocking=True)

            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in target_batch]).to(device)
            labels = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in target_batch]).to(device)
            class_weights = torch.ones(labels.shape).to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            logits = outputs.logits

            if self.multiclass:
                labels = labels.long()
                logits = torch.softmax(logits.reshape(logits.size(0), -1, self.multiclass_num_classes), dim=-1) # batch size x num targets x num classes per target
                loss = torch.cat([loss_func(logits[:, target_index, :], labels[:, target_index]).unsqueeze(1) for target_index in range(logits.size(1))], dim=1) * class_weights * mask
            else:
                loss = loss_func(logits, labels) * class_weights * mask

            loss = loss.sum() / mask.sum()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(cur_epoch=epoch, cur_step=i)

            y_preds.extend(logits.data.cpu().numpy().tolist())
            y_labels.extend(target_batch)
            loss_all += loss.item() * labels.size()[0]

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type)

        results["loss"] = loss_all / len(data_loader.dataset)
        results["epoch"] = epoch
        return results

    def evaluation(self, model, data_loader, loss_func, scaler=None, device="cpu"):
        model.eval()

        loss_all = 0
        y_preds, y_labels = [], []
        for _, batch_data in enumerate(data_loader):
            target_batch = batch_data["targets"]
            for k, v in batch_data.items():
                if type(v) == torch.Tensor or issubclass(type(v), Batch):
                    batch_data[k] = v.to(device, non_blocking=True)

            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in target_batch]).to(device)
            labels = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in target_batch]).to(device)
            class_weights = torch.ones(labels.shape).to(device)

            outputs = model(batch_data)
            logits = outputs.logits

            # Inverse scale if regression
            if self.regression and scaler is not None:
                logits = torch.FloatTensor(scaler.inverse_transform(logits.detach().cpu().numpy())).to(device)

            if self.multiclass:
                labels = labels.long()
                logits = torch.softmax(logits.reshape(logits.size(0), -1, self.multiclass_num_classes), dim=-1)
                loss = torch.cat([loss_func(logits[:, target_index, :], labels[:, target_index]).unsqueeze(1) for target_index in range(logits.size(1))], dim=1) * class_weights * mask
            else:
                loss = loss_func(logits, labels) * class_weights * mask
            loss = loss.sum() / mask.sum()

            if self.classification:
                logits = torch.sigmoid(logits)
            y_preds.extend(logits.data.cpu().numpy().tolist())
            y_labels.extend(target_batch)
            loss_all += loss.item() * labels.size()[0]

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type)

        results["loss"] = loss_all / len(data_loader.dataset)
        results["predictions"] = y_preds
        results["targets"] = y_labels
        return results

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
                print('valid target len is 0')
                assert False

        if not isinstance(metric_type, list):
            results = {metric_type: []}
        else:
            results = {metric_t: [] for metric_t in metric_type}

        for i in range(num_tasks):
            # # Skip if all targets or preds are identical, otherwise we'll crash during classification
            if task_type == 'classification':
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



