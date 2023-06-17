#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy.
 All rights reserved.

 @author: Jiahua Rao, Jiancong Xie, Junjie Xie
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


import os
import numpy as np
import collections

from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import torch
from torch_geometric.data import Batch

from molrep.common.registry import registry
from molrep.tasks.base_task import BaseTask
from molrep.models.metrics import get_metric
from molrep.explainer import attribution_metric as att_metrics
from molrep.common.explainer_utils import *


@registry.register_task("molecular_explainer")
class ExplainerTask(BaseTask):
    def __init__(self, task_type, num_tasks, metric_type, multiclass_num_classes=1, attribution_truth=None):
        super().__init__()

        self.task_type = task_type
        self.num_tasks = num_tasks
        self.metric_type = metric_type

        self.classification = self.task_type == 'classification'
        self.multiclass = self.task_type == 'multiclass-classification'
        self.regression = self.task_type == 'regression'
        self.multiclass_num_classes = multiclass_num_classes
        assert not (self.classification and self.regression and self.multiclass)

        self.attribution_truth = attribution_truth
        if attribution_truth is not None:
            self.attribution_truth = self.get_attribution_truth(attribution_truth)

    @classmethod
    def setup_task(cls, cfg):
        task_type = cfg.datasets_cfg.task_type
        num_tasks = cfg.datasets_cfg.num_tasks
        metric_type = cfg.datasets_cfg.metric_type
        multiclass_num_classes = cfg.datasets_cfg.multiclass_num_classes
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
            outputs = model(batch_data)
            logits = outputs.logits

            if self.multiclass:
                labels = labels.long()
                logits = torch.softmax(logits.reshape(logits.size(0), -1, self.multiclass_num_classes), dim=-1)
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

    def evaluation(self, model, explainer, data_loader, loss_func, scaler=None, device="cpu", **kwargs):
        model.eval()
        eval_explainer = kwargs.get("eval_explainer", False)
        is_testing = kwargs.get("is_testing", False)

        loss_all = 0
        y_preds, y_labels = [], []
        atom_attr_preds, bond_attr_preds = [], []
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

            if is_testing:
                atom_attr, bond_attr = explainer.explain(batch_data, model)
                if atom_attr is not None:
                    # new_atom_attr = model.unbatch(atom_attr, batch_data, is_atom=True)
                    atom_attr_preds.extend(atom_attr)

                if bond_attr is not None:
                    # new_bond_attr = model.unbatch(bond_attr, batch_data, is_atom=False)
                    bond_attr_preds.extend(bond_attr)

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

            y_preds.extend(logits.data.cpu().numpy().tolist())
            y_labels.extend(target_batch)
            loss_all += loss.item() * labels.size()[0]

        atom_importance = atom_attr_preds if len(atom_attr_preds) > 0 else None
        bond_importance = bond_attr_preds if len(bond_attr_preds) > 0 else None

        results = self.evaluate_predictions(preds=y_preds, targets=y_labels,
                                            num_tasks=self.num_tasks, metric_type=self.metric_type,
                                            task_type=self.task_type)

        results["loss"] = loss_all / len(data_loader.dataset)
        results["predictions"] = y_preds
        results["targets"] = y_labels

        if eval_explainer and self.attribution_truth is not None:
            attribution_results, opt_threshold = self.evaluate_attributions(data_loader.dataset, atom_importance, bond_importance)
            results["attribution_results"] = attribution_results
            results["opt_threshold"] = opt_threshold

        results["atom_importance"], results["bond_importance"] = atom_importance, bond_importance
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

    def visualization(self, dataset, atom_importance, bond_importance, threshold=1e-4, use_negative=False, set_weights=False, svg_dir=None, vis_factor=1.0, img_width=400, img_height=200, drawAtomIndices=False):
        svg_list = []

        smiles_list = [d.smiles for d in dataset]
        att_probs = preprocessing_attributions(smiles_list, atom_importance, bond_importance, normalizer='MinMaxScaler')

        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            cp = Chem.Mol(mol)
            atom_imp = att_probs[idx]

            highlightAtomColors, cp = determine_atom_col(cp, atom_imp, eps=threshold, use_negative=use_negative, set_weights=set_weights)
            highlightAtoms = list(highlightAtomColors.keys())

            highlightBondColors = determine_bond_col(highlightAtomColors, mol)
            highlightBonds = list(highlightBondColors.keys())

            highlightAtomRadii = {
                # k: np.abs(v) * vis_factor for k, v in enumerate(atom_imp)
                k: 0.1 * vis_factor for k, v in enumerate(atom_imp)
            }

            rdDepictor.Compute2DCoords(cp, canonOrient=True)
            drawer = rdMolDraw2D.MolDraw2DCairo(img_width, img_height)
            if drawAtomIndices:
                drawer.drawOptions().addAtomIndices = True
            drawer.drawOptions().useBWAtomPalette()
            drawer.DrawMolecule(
                cp,
                highlightAtoms=highlightAtoms,
                highlightAtomColors=highlightAtomColors,
                # highlightAtomRadii=highlightAtomRadii,
                highlightBonds=highlightBonds,
                highlightBondColors=highlightBondColors,
            )
            drawer.FinishDrawing()
            drawer.WriteDrawingText(os.path.join(svg_dir, f"{idx}.png"))
            svg = drawer.GetDrawingText()#.replace("svg:", "")
            svg_list.append(svg)
        return svg_list


    def evaluate_attributions(self, dataset, atom_importance, bond_importance, binary=False, other=None):
        att_true = self.attribution_truth

        stats = collections.OrderedDict()
        smiles_list = [d.smiles for d in dataset]
        att_probs = preprocessing_attributions(smiles_list, atom_importance, bond_importance)
        
        if binary:
            opt_threshold = -1
            stats['ATT F1'] = np.nanmean(
                att_metrics.attribution_f1(att_true, att_probs))
            stats['ATT ACC'] = np.nanmean(
                att_metrics.attribution_accuracy(att_true, att_probs))
        else:
            opt_threshold = att_metrics.get_optimal_threshold(att_true, att_probs)
            # opt_threshold = 0.5
            att_binary = [np.array([1 if att>opt_threshold else 0 for att in att_prob]) for att_prob in att_probs]

            stats['Attribution AUROC'] = np.nanmean(
                att_metrics.attribution_auroc(att_true, att_probs))
            stats['Attribution F1'] = np.nanmean(
                att_metrics.attribution_f1(att_true, att_binary))
            stats['Attribution ACC'] = np.nanmean(
                att_metrics.attribution_accuracy(att_true, att_binary))
            stats['Attribution Precision'] = np.nanmean(
                att_metrics.attribution_precision(att_true, att_binary))
            stats['Attribution AUROC Mean'] = att_metrics.attribution_auroc_mean(att_true, att_probs)
            stats['Attribution ACC Mean'] = att_metrics.attribution_accuracy_mean(att_true, att_binary)

        return stats, opt_threshold

    def evaluate_cliffs(self, dataset, atom_importance, bond_importance):
        smiles_list = [d.smiles for d in dataset]
        att_true_pair = self.attribution_truth
        att_probs = preprocessing_attributions(smiles_list, atom_importance, bond_importance, normalizer='MaxAbsScaler')
        att_probs_dict = dict(zip(smiles_list, att_probs))
        
        att_probs_reset, att_true = [], []
        smiles_list = list(smiles_list)
        for idx in range(len(att_true_pair)):
            smiles_1 = att_true_pair[idx]['SMILES_1']
            smiles_2 = att_true_pair[idx]['SMILES_2']

            att_probs_reset.append(att_probs_dict[smiles_1])
            att_true_1 = att_true_pair[idx]['attribution_1']
            att_true.append(att_true_1)

            att_probs_reset.append(att_probs_dict[smiles_2])
            att_true_2 = att_true_pair[idx]['attribution_2']
            att_true.append(att_true_2)

        opt_threshold = att_metrics.get_optimal_threshold(att_true, att_probs_reset, multi=True)
        att_binary = [np.array([1 if att>opt_threshold else -1 if att<(-opt_threshold) else 0 for att in att_prob]) for att_prob in att_probs_reset]

        stats = collections.OrderedDict()
        stats['ATT F1'] = np.nanmean(
            att_metrics.attribution_f1(att_true, att_binary))
        stats['ATT ACC'] = np.nanmean(
            att_metrics.attribution_accuracy(att_true, att_binary))
        return stats, opt_threshold

    def get_attribution_truth(self, path):
        attribution = np.load(path, allow_pickle=True)['attributions']
        return [attribution[idx]['node_atts'] for idx in range(len(attribution))]
