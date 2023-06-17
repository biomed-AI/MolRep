"""
 Copyright (c) 2022, Sun Yat-sen Univeristy.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import molrep.tasks as tasks
from molrep.common.config import Config
from molrep.common.registry import registry


def now():
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--job-id", type=str, required=False, default='now', help=".")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args


def get_experiments_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    experiment_cls = registry.get_experiment_class(cfg.run_cfg.get("task", "property_prediction"))
    return experiment_cls


def setup_seeds(config):
    seed = config.run_cfg.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    job_id = args.job_id if args.job_id != 'now' else now()

    task = tasks.setup_task(cfg)
    dataset, scaler = task.build_datasets(cfg)
    model = task.build_model(cfg)

    experiment = get_experiments_class(cfg)(
        cfg=cfg, task=task, datasets=dataset, scaler=scaler, model=model, job_id=job_id,
    )
    experiment.train()


if __name__ == "__main__":
    main()