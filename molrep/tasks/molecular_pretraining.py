#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy.
 All rights reserved.

 @author: Jiahua Rao, Jiancong Xie, Junjie Xie
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


import torch
from torch_geometric.data import Data

from molrep.common.dist_utils import *
from molrep.common.registry import registry
from molrep.tasks.base_task import BaseTask


@registry.register_task("molecular_pretraining")
class PretraingTask(BaseTask):
    def __init__(self, pretrain_losses):
        super().__init__()
        self.pretrain_losses = pretrain_losses

    @classmethod
    def setup_task(cls, cfg):
        pretrain_losses = cfg.model_cfg.get("pretrain_losses", [])

        return cls(
            pretrain_losses=pretrain_losses,
        )

    def build_datasets(self, cfg):
        return super().build_datasets(cfg)

    def train_step(self, model, samples):
        outputs = model(samples)
        return outputs

    def train_epoch(
            self,
            epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            grad_scaler=None,
            log_freq=50,
            accum_grad_iters=1,
            use_distributed=False,
            device="cpu"
    ):
        metric_logger = MetricLogger(delimiter="  ")
        for name in self.pretrain_losses:
            metric_logger.add_meter(name, SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        header = "Train: data epoch: [{}]".format(epoch)
        use_amp = grad_scaler is not None

        if use_distributed:
            data_loader.sampler.set_epoch(self.cur_epoch)

        for i, batch_data in enumerate(metric_logger.log_every(data_loader, log_freq, header)):

            lr_scheduler.step(cur_epoch=epoch, cur_step=i)
            for k, v in batch_data.items():
                if type(v) == torch.Tensor or issubclass(type(v), Data):
                    batch_data[k] = v.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = self.train_step(model, batch_data)
                total_loss = sum([v for k, v in outputs.items() if "loss" in k])

            # after_train_step()
            if use_amp:
                grad_scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=total_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            kwargs = {name: outputs[name].item() for name in self.pretrain_losses}
            metric_logger.update(**kwargs)

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass