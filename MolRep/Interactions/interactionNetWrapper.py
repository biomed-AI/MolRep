# -*- coding: utf-8 -*-
"""
Created on 2022.06.20

@author: Jiahua Rao

"""
import time
from datetime import timedelta

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from MolRep.Interactions.interactionEvaluator import InteractionEvaluator


def format_time(avg_time):
    avg_time = timedelta(seconds=avg_time)
    total_seconds = int(avg_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(avg_time.microseconds)[:3]}"



class InteractionNetWrapper:
    
    def __init__(self, model, dataset_configs, model_config):
        self.model = model
        self.model_name = model_config.exp_name

        self.task_type = dataset_configs["task_type"]
        # 
        
        self.dataset_name = dataset_configs["name"]
        self.num_epochs = model_config['num_epochs']

        self.device = torch.device(model_config['device'])

        self.dataset_configs = dataset_configs
        self.model_config = model_config

        self.evaluator = InteractionEvaluator(self.dataset_configs)
        self.metric_type = dataset_configs["eval_metric"]
        self.print_metric_type = self.metric_type[0] if isinstance(self.metric_type, list) else self.metric_type


    def train(self, inputs, log_every=10, logger=None, other=None):

        time_per_epoch = []
        
        best_val_res = 0.0
        best_params = None
        
        if self.model_name == 'CFLP':
            pretrained_params = self.model.pretrain(inputs, logger)
            z = self.model.obtained_z_from_encoder(inputs, pretrained_params)
            other = {'z': z}

        # Training for Deep learning Methods
        for i in range(1, self.num_epochs+1):
            begin = time.time()
            train_loss = self.model.train(inputs=inputs, other=other)
            end = time.time()
            duration = end - begin
            time_per_epoch.append(end)

            valid_loss, test_loss, (valid_results, test_results) = self.model.test(inputs=inputs,
                                                                                   evaluator=self.evaluator.evaluator,
                                                                                   other=other)
            
            if i % log_every == 0 or i == 1:
                msg = [
                    f"Epoch: {i}",
                    f"train loss: {train_loss:03f}",
                    f"valid loss: {valid_loss:03f}",
                    f"test loss: {test_loss:03f}" 
                ]
                logger.log(', '.join(msg))
                msg = ["[VALID]"]
                for k, v in valid_results.items():
                    msg.append(f"valid_{k}: {v:04f}")
                logger.log(', '.join(msg))
                msg = ["[TEST]"]
                for k, v in test_results.items():
                    msg.append(f"test_{k}: {v:04f}")
                logger.log(', '.join(msg))

                logger.log(f"- Elapsed time: {str(duration)[:4]}s , Time estimation in a fold: {str(duration*self.num_epochs/60)[:4]}min")

            if valid_results[self.print_metric_type] > best_val_res:
                best_val_res = valid_results[self.print_metric_type]
                best_params = parameters_to_vector(self.model.parameters())
                test_results['best_val'] = valid_results[self.print_metric_type]

        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())

        elapsed = format_time(avg_time_per_epoch)
        if best_params is not None:
            vector_to_parameters(best_params, self.model.parameters())
        return train_loss, valid_results, test_results, elapsed

        
    def test(self, inputs, logger=None):
        
        test_eval_metric = self.dataset_configs['eval_metric']
        results = self.model.test(inputs)
        
        logger.log(f'[TEST] test %s: %.6f' % (test_eval_metric, results[test_eval_metric][1]))

        return results