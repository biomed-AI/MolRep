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


@registry.register_task("molecular_explainer")
class ExplainerTask(BaseTask):
    def __init__(self, task_type, num_tasks, metric_type):
        super().__init__()

        self.task_type = task_type
        self.num_tasks = num_tasks
        self.metric_type = metric_type

    @classmethod
    def setup_task(cls, cfg):
        task_type = cfg.datasets_cfg.task_type
        num_tasks = cfg.datasets_cfg.num_tasks
        metric_type = cfg.datasets_cfg.metric_type
        return cls(
            task_type = task_type,
            num_tasks = num_tasks,
            metric_type = metric_type,
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
            scaler=None,
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
            output = model(batch_data)
            if not isinstance(output, tuple):
                output = (output,)

            # Inverse scale if regression
            if scaler is not None:
                output = list(output)
                output[0] = torch.Tensor(scaler.inverse_transform(output[0].detach().cpu().numpy()))
                output = tuple(output)

            if self.task_type == 'Multi-Classification':
                labels = labels.long()
                loss = torch.cat([loss_func(labels[:, target_index], output[0][:, target_index, :]).unsqueeze(1) for target_index in range(output[0].size(1))], dim=1) * class_weights * mask
            else:
                loss = loss_func(labels, *output) * class_weights * mask

            loss = loss.sum() / mask.sum()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(cur_epoch=epoch, cur_step=i)

            y_preds.extend(output[0].data.cpu().numpy().tolist())
            y_labels.extend(target_batch)
            loss_all += loss.item() * labels.size()[0]

        if self.task_type != 'Regression':
            y_preds = torch.sigmoid(torch.FloatTensor(y_preds))

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type)

        results["loss"] = loss_all / len(data_loader.dataset)
        results["epoch"] = epoch
        return results

    def evaluation(self, model, explainer, data_loader, loss_func, scaler=None, device="cpu"):
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

            output = model(batch_data)
            if not isinstance(output, tuple):
                output = (output,)

            # Inverse scale if regression
            if scaler is not None:
                output = list(output)
                output[0] = torch.Tensor(scaler.inverse_transform(output[0].detach().cpu().numpy()))
                output = tuple(output)

            if self.task_type == 'Multi-Classification':
                labels = labels.long()
                loss = torch.cat([loss_func(labels[:, target_index], output[0][:, target_index, :]).unsqueeze(1) for target_index in range(output[0].size(1))], dim=1) * class_weights * mask
            else:
                loss = loss_func(labels, *output) * class_weights * mask
            loss = loss.sum() / mask.sum()

            y_preds.extend(output[0].data.cpu().numpy().tolist())
            y_labels.extend(target_batch)
            loss_all += loss.item() * labels.size()[0]

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type)

        results["loss"] = loss_all / len(data_loader.dataset)
        results["predictions"] = y_preds
        results["targets"] = y_labels
        return results




