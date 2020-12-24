# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
"""

import numpy as np

from xgboost import XGBClassifier, XGBRegressor

class XGboost:
    """docstring for XGboost"""
    def __init__(self, model_configs, task_type='Regression'):
        self.model_configs = model_configs

        self.max_depth = model_configs['max_depth']
        self.learning_rate = model_configs['learning_rate']
        self.n_estimators = model_configs['n_estimators']
        self.objective = model_configs['objective']
        self.booster = model_configs['booster']
        self.subsample = model_configs['subsample']
        self.colsample_bylevel = model_configs['colsample_bylevel']
        self.colsample_bytree = model_configs['colsample_bytree']
        self.min_child_weight = model_configs['min_child_weight']
        self.reg_alpha = model_configs['reg_alpha']
        self.reg_lambda = model_configs['reg_lambda']
        self.scale_pos_weight = model_configs['scale_pos_weight']
        self.max_delta_step = model_configs['max_delta_step']

        self.random_seed = model_configs['random_seed']
        self.eval_metric = model_configs['early_stopping']['eval_metric']
        self.early_stopping_round = model_configs['early_stopping']['round']

        self.task_type = task_type
        np.random.seed(seed=self.random_seed)

        self.setup_model()

    def setup_model(self):
        if self.task_type == 'Classification':
            self.model = XGBClassifier(
                            max_depth=self.max_depth,
                            learning_rate=self.learning_rate,
                            n_estimators=self.n_estimators,
                            objective=self.objective,
                            booster=self.booster,
                            subsample=self.subsample,
                            colsample_bylevel=self.colsample_bylevel,
                            colsample_bytree=self.colsample_bytree,
                            min_child_weight=self.min_child_weight,
                            reg_alpha=self.reg_alpha,
                            reg_lambda=self.reg_lambda,
                            scale_pos_weight=self.scale_pos_weight,
                            max_delta_step=self.max_delta_step,
                            random_state=self.random_seed,
                            silent=False,
                            n_jobs=8
                        )

        elif self.task_type == 'Regression':
            self.model = XGBRegressor(
                            max_depth=self.max_depth,
                            learning_rate=self.learning_rate,
                            n_estimators=self.n_estimators,
                            objective=self.objective,
                            booster=self.booster,
                            subsample=self.subsample,
                            colsample_bylevel=self.colsample_bylevel,
                            colsample_bytree=self.colsample_bytree,
                            min_child_weight=self.min_child_weight,
                            reg_alpha=self.reg_alpha,
                            reg_lambda=self.reg_lambda,
                            scale_pos_weight=self.scale_pos_weight,
                            random_state=self.random_seed,
                            silent=False,
                            n_jobs=8
                        )

        else:
            raise "Task type Error!"


    def fit_model(self, train_loader):
        self.model.fit(train_loader[0], train_loader[1])


    def predict(self, test_loader):
        if self.task_type == 'Classification':
            return self.model.predict_proba(test_loader[0])
        else:
            return self.model.predict(test_loader[0])