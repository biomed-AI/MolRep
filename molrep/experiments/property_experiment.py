

import os, datetime

from molrep.common.utils import *
from molrep.common.registry import registry
from molrep.experiments.experiment import Experiment

@registry.register_experiment("property_prediction")
class PropertyExperiment(Experiment):
    """
    Experiment provides a layer of abstraction to avoid that all models implement the same interface
    """

    def __init__(self, cfg, task, model, datasets, job_id, scaler=None):
        super().__init__(
            cfg=cfg, task=task, datasets=datasets, scaler=scaler, model=model, job_id=job_id,
        )

        self.test_every_epochs = self.config.run_cfg.get("type", None) == 'train_test'

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = -1

        self.log_config()
        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        print("Start training")
        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                print(f"Training on Epoch:{cur_epoch}")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    print("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    
                    agg_metrics = val_log[self.metric_type[0]]
                    if compare_metrics(agg_metrics, best_agg_metric, self.metric_type[0]) and split_name == "val":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics
                        self._save_checkpoint(cur_epoch, is_best=True)
                        self._save_test_results({"val": val_log})

                    val_log.update({"best_epoch": best_epoch})
                    self.log_stats(val_log, split_name)

            elif len(self.test_splits) > 0 and self.test_every_epochs:
                for split_name in self.test_splits:
                    print("Evaluating on {}.".format(split_name))

                    test_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    
                    agg_metrics = test_log[self.metric_type[0]]
                    if compare_metrics(agg_metrics, best_agg_metric, self.metric_type[0]) and split_name == "test":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics
                        self._save_checkpoint(cur_epoch, is_best=True)
                        self._save_test_results({"test": test_log})

                    test_log.update({"best_epoch": best_epoch})
                    self.log_stats(test_log, split_name)

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

        # evaluate phase: evaluate
        test_epoch = "best" if best_epoch >= 0 else cur_epoch
        test_results = self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)
        self._save_test_results(test_results)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()
        print(f"Loading trained model from Epoch: {cur_epoch}\n")

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
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
        return self.task.evaluation(self.model, data_loader, loss_func=self.loss_func, scaler=self.scaler, device=self.device)

