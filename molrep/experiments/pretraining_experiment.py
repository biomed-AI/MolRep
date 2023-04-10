

import os, datetime

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from molrep.common.utils import *
from molrep.common.registry import registry
from molrep.common.logger import Logger
from molrep.experiments.experiment import Experiment

from molrep.common.dist_utils import (
    is_main_process,
    main_process,
    get_rank,
    get_world_size,
)


@registry.register_experiment("molecular_pretraining")
class PretrainingExperiment(Experiment):
    """
    Experiment provides a layer of abstraction to avoid that all models implement the same interface
    """

    def __init__(self, cfg, task, model, datasets, job_id, scaler=None):
        super().__init__(
            cfg=cfg, task=task, datasets=datasets, scaler=scaler, model=model, job_id=job_id,
        )
        self._grad_scaler = None

    @property
    def use_distributed(self):
        return self.config.run_cfg.get("distributed", True)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
    ):
        """
        Create dataloaders for training and validation.
        """
        pretrain_model_name = self.config.model_cfg.name
        kwargs = {"pretrain_model_name": pretrain_model_name}

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            # map-style dataset are concatenated together
            # setup distributed sampler
            if self.use_distributed:
                sampler = DistributedSampler(
                    dataset,
                    shuffle=is_train,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                )
            else:
                sampler = None

            loader = DataLoader(
                dataset,
                batch_size=bsz,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=sampler is None and is_train,
                collate_fn=lambda data_list: collate_fn(data_list, **kwargs),
                drop_last=True if is_train else False,
            )

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)
            loaders.append(loader)
        return loaders

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)
            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DistributedDataParallel(
                        self._model, device_ids=[self.config.run_cfg.gpu]
                    )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def grad_scaler(self):
        amp = self.config.run_cfg.get("amp", False)
        if amp:
            if self._grad_scaler is None:
                self._grad_scaler = torch.cuda.amp.GradScaler()
        return self._grad_scaler

    def train(self):
        start_time = time.time()
        self.log_config()

        # resume from checkpoint if specified
        if self.evaluate_only or self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path, evaluate_only=self.evaluate_only)

        print("Start training.")
        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # pre-training phase
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(split_name="train", stats=train_stats)
            # we just save the checkpoint at the end of each epoch.
            self._save_checkpoint(cur_epoch, is_best=False)
            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        pass

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            grad_scaler=self.grad_scaler,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
            use_distributed=self.use_distributed,
            device=self.device,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        pass

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        save_obj = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "grad_scaler": self.grad_scaler.state_dict() if self.grad_scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.result_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items() if k in ["epoch", "loss", "best_epoch"] + self.metric_type}}
            self.logger.log(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        self.logger.log(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @main_process
    def setup_logger(self):
        LOGGER_BASE = os.path.join(self.output_dir, "logger")
        Path(LOGGER_BASE).mkdir(parents=True, exist_ok=True)
        self.logger = Logger(str(os.path.join(LOGGER_BASE, f"logging.log")), mode='a')
