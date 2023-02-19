

import os
import time
import datetime

import torch

from molrep.common.registry import registry
from molrep.experiments.experiment import Experiment


@registry.register_experiment("molecular_explainer")
class ExplainerExperiment(Experiment):
    """
    Experiment provides a layer of abstraction to avoid that all models implement the same interface
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(
            cfg=cfg, task=task, datasets=datasets, model=model, job_id=job_id,
        )
        self._explainer = None

    def test_loader(self, split_name):
        return self.dataloaders[split_name]

    @property
    def explainer(self):
        """
        A property to get the explainer model on the device.
        """
        if self._explainer is None:
            explainer_cls = registry.get_explainer_class(self.config.run_cfg.get("explainer", None).name.lower())
            self._explainer = explainer_cls(self.config.explainer_cfg)
        return self._explainer

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()
        # resume from checkpoint if specified
        if self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path, evaluate_only=self.evaluate_only)

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
                        split_name=split_name, cur_epoch=cur_epoch, eval_explainer=False,
                    )
                    
                    agg_metrics = val_log[self.metric_type[0]]
                    if agg_metrics > best_agg_metric and split_name == "val":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics

                        self._save_checkpoint(cur_epoch, is_best=True)

                    val_log.update({"best_epoch": best_epoch})
                    self.log_stats(val_log, split_name)

            else:
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False)

        # evaluate phase: test & explainer
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
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
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload, eval_explainer=True,
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

    def eval_epoch(self, split_name, cur_epoch, skip_reload=False, eval_explainer=True):
        """
        Evaluate and Explain the model on a given split.
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
        kwargs = {"eval_explainer": eval_explainer, "is_testing": split_name in self.test_splits}
        return self.task.evaluation(self.model, self.explainer, data_loader, loss_func=self.loss_func, scaler=self.feature_scaler, device=self.device, **kwargs)

    def _save_test_results(self, test_results):
        """
        Save the test results at the Final evaluation.
        """
        for k, result in test_results.items():
            print("Evaluating on {}.".format(k))
            self.log_stats(result, k)

            save_obj = {
                "predictions": result['predictions'],
                "targets": result['targets']
            }
            save_to = os.path.join(self.result_dir, f"{k}_predictions.pth")
            print("Saving predictions to {}.".format(save_to))
            torch.save(save_obj, save_to)

            save_obj = {
                "atom_importance": result['atom_importance'],
                "bond_importance": result['bond_importance']
            }
            save_to = os.path.join(self.result_dir, f"{k}_explainer_predictions.pth")
            print("Saving explainer predictions to {}.".format(save_to))
            torch.save(save_obj, save_to)

            save_to = os.path.join(self.result_dir, "svg")
            os.makedirs(save_to, exist_ok=True)
            self.task.visualization(
                dataset=self.test_loader(split_name=k).dataset,
                atom_importance=result['atom_importance'],
                bond_importance=result['bond_importance'],
                svg_dir=save_to
            )


