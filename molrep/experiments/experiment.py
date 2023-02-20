

import os
import datetime

from torch.utils.data import DataLoader, DistributedSampler

from molrep.common.utils import *
from molrep.common.registry import registry
from molrep.models.losses import get_loss_func
from molrep.common.logger import Logger
from molrep.data.datasets.base_dataset import MoleculeSampler

from molrep.common.dist_utils import (
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)

@registry.register_experiment("base")
class Experiment:
    """
    Experiment provides a layer of abstraction to avoid that all models implement the same interface
    """

    def __init__(self, cfg, task, model, datasets, job_id, scaler=None):

        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets
        self._scaler = scaler

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._feature_scaler = None
        self._dataloaders = None
        self._lr_sched = None
        self._loss_func = None

        self.start_epoch = 0
        self.use_distributed = False

        # self.setup_seeds()
        self.setup_output_dir()
        self.setup_logger()

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

        return self._model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=float(self.config.run_cfg.get("init_lr", 0.0001)),
                weight_decay=float(self.config.run_cfg.get("weight_decay", 0)),
            )

        return self._optimizer

    @property
    def loss_func(self):
        if self._loss_func is None:
            self._loss_func = get_loss_func(self.config.datasets_cfg.task_type, self.config.model_cfg.arch)
        return self._loss_func

    @property
    def scaler(self):
        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            max_epoch = self.max_epoch
            min_lr = self.min_lr
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
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size
                if split == "train"
                else self.config.run_cfg.get("batch_size_eval", self.config.run_cfg.batch_size)
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                collate_fns.append(getattr(dataset, "collate_fn", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.get("num_workers", 0),
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
            )
            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def metric_type(self):
        _metric_type = self.config.datasets_cfg.metric_type
        return [str(_metric_type)] if isinstance(_metric_type, str) else list(_metric_type)

    @property
    def max_epoch(self):
        return self.config.run_cfg.get("num_epochs", 1)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.get("evaluate_only", False)

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", ["train"])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def valid_splits(self):
        if "val" not in self.datasets.keys():
            return []
        return ['val']

    @property
    def test_splits(self):
        return ['test']

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("repo_root"))

        output_dir = lib_root / "outputs" / self.config.run_cfg.get("task", "property_prediction")
        output_dir = output_dir / f"{self.config.model_cfg.arch}_{self.config.datasets_cfg.name}"
        output_dir = output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()
        # resume from checkpoint if specified
        if self.evaluate_only or self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path, evaluate_only=self.evaluate_only)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                print("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    print("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch,
                    )

                    agg_metrics = val_log["agg_metrics"]
                    if compare_metrics(agg_metrics, best_agg_metric, self.metric_type[0]) and split_name == "val":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics
                        self._save_checkpoint(cur_epoch, is_best=True)

                    val_log.update({"best_epoch": best_epoch})
                    self.log_stats(val_log, split_name)

            else:
                self._save_checkpoint(cur_epoch, is_best=False)

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload,
                )

                print("Evaluating on {}.".format(split_name))
                self.log_stats(test_logs[split_name], split_name)

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
            scaler=self._scaler,
            device=self.device,
        )

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
        return self.task.evaluation(self.model, data_loader, scaler=self._scaler, device=self.device)

    def _load_checkpoint(self, url_or_filename, evaluate_only=False):
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
        if self.scaler and "data_scaler" in checkpoint:
            self.scaler = StandardScaler(checkpoint['data_scaler']['means'],
                                         checkpoint['data_scaler']['stds']) if checkpoint['data_scaler'] is not None else None

        if not evaluate_only:
            self.start_epoch = checkpoint["epoch"] + 1

        print("Resume checkpoint from {}".format(url_or_filename))

    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        save_obj = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            'data_scaler': {
                'means': self.scaler.means,
                'stds': self.scaler.stds
            } if self.scaler is not None else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.result_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _save_test_results(self, test_results):
        """
        Save the test results at the Final evaluation.
        """

        for k, result in test_results.items():
            save_obj = {
                "predictions": result['predictions'],
                "targets": result['targets']
            }
            save_to = os.path.join(self.result_dir, f"{k}_predictions.pth")
            print("Saving {} predictions to {}.".format(k, save_to))
            torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.result_dir, "checkpoint_best.pth")

        print("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        return model


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
        class_balance = self.config.run_cfg.get("class_balance", False)
        seed = self.config.run_cfg.get("seed", 42)
        follow_batch = self.config.run_cfg.get("follow_batch", [])
        atom_messages = self.config.model_cfg.get("atom_messages", False)
        kwargs = {"atom_messages": atom_messages, "follow_batch": follow_batch}

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
                if not self.use_dist_eval_sampler:
                    # e.g. retrieval evaluation
                    sampler = sampler if is_train else None
            else:
                # sampler = None
                sampler = MoleculeSampler(
                    dataset=dataset,
                    class_balance=class_balance,
                    shuffle=is_train,
                    seed=seed
                )

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

    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items() if k in ["epoch", "loss", "best_epoch"] + self.metric_type}}
            self.logger.log(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    def log_config(self):
        self.logger.log(json.dumps(self.config.to_dict(), indent=4) + "\n")

    def setup_logger(self):
        LOGGER_BASE = os.path.join(self.output_dir, "logger")
        Path(LOGGER_BASE).mkdir(parents=True, exist_ok=True)
        self.logger = Logger(str(os.path.join(LOGGER_BASE, f"logging.log")), mode='a')