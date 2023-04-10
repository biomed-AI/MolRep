#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy, inc.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


from molrep.common.registry import registry

from molrep.tasks.base_task import BaseTask
from molrep.tasks.property_prediction import PropertyTask
from molrep.tasks.molecular_explainer import ExplainerTask
from molrep.tasks.molecular_pretraining import PretraingTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    'BaseTask',
    'PropertyTask',
    'ExplainerTask',
    'PretraingTask'
]