

import os, datetime

from molrep.common.utils import *
from molrep.common.registry import registry
from molrep.models.losses import get_loss_func
from molrep.experiments.experiment import Experiment

@registry.register_experiment("property_prediction")
class ProPertyExperiment(Experiment):
    """
    Experiment provides a layer of abstraction to avoid that all models implement the same interface
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(
            cfg=cfg, task=task, datasets=datasets, model=model, job_id=job_id,
        )

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def model(self):
        """
        A property to get the wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)
            self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
            )

        return self._optimizer

    @property
    def loss_func(self):
        if self._loss_func is None:
            self._loss_func = get_loss_func(self.config.datasets_cfg.task_type, self.config.model_cfg.arch)
        return self._loss_func

    @property
    def feature_scaler(self):
        if self._feature_scaler is None:
            use_feature_scaler = self.config.run_cfg.get("feature_scaler", None)
            if use_feature_scaler is not None:
                self._feature_scaler = StandardScaler()

        return self._feature_scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", 1)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """

        if self._dataloaders is None:
            # create dataloaders
            split_names = sorted(self.datasets.keys())
            datasets = [self.datasets[split] for split in split_names]
            self._dataloaders = {k: v for k, v in zip(split_names, datasets)}

        return self._dataloaders

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()
        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        self.logger.log("Start training")
        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                self.logger.log(f"Training on Epoch:{cur_epoch}")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    self.logger.log("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    
                    agg_metrics = val_log[self.metric_type]
                    if agg_metrics > best_agg_metric and split_name == "val":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics

                        self._save_checkpoint(cur_epoch, is_best=True)

                    val_log.update({"best_epoch": best_epoch})
                    self.log_stats(val_log, split_name)

            else:
                self._save_checkpoint(cur_epoch, is_best=False)

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        test_results = self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)
        self._save_test_results(test_results)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.log("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()
        self.logger.log(f"Model from Epoch: {cur_epoch}\n")

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            loss_func=self.loss_func,
            scaler=self.feature_scaler,
            device=self.device,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.
        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        model = self.model
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)

        model.eval()
        return self.task.evaluation(self.model, data_loader, loss_func=self.loss_func, scaler=self.feature_scaler, device=self.device)


    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.model.load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.feature_scaler and "scaler" in checkpoint:
            self.feature_scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.logger.log("Resume checkpoint from {}".format(url_or_filename))

    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        save_obj = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.feature_scaler.state_dict() if self.feature_scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.result_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        self.logger.log("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.result_dir, "checkpoint_best.pth")

        self.logger.log("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        return model
