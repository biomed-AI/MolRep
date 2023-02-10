# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForest:
    """docstring for RandomForest"""
    def __init__(self, model_configs, task_type='Regression', logger=None):
        self.model_configs = model_configs

        self.max_features = model_configs['max_features']
        self.n_estimators = model_configs['n_estimators']
        self.min_samples_leaf = model_configs['min_samples_leaf']
        self.class_weight = model_configs['class_weight']
        self.random_seed = model_configs['random_seed']

        self.task_type = task_type
        np.random.seed(seed=self.random_seed)
        self.logger = logger

        self.setup_model()

    def setup_model(self):
        if self.task_type == 'Classification':
            self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                                max_features=self.max_features,
                                                min_samples_leaf=self.min_samples_leaf,
                                                n_jobs=8,
                                                class_weight=self.class_weight,
                                                random_state=self.random_seed,
                                                oob_score=False,
                                                verbose=1)

        elif self.task_type == 'Regression':
            self.model = RandomForestRegressor(n_estimators=self.n_estimators,
                                               max_features=self.max_features,
                                               min_samples_leaf=self.min_samples_leaf,
                                               n_jobs=8,
                                               random_state=self.random_seed,
                                               oob_score=False,
                                               verbose=1)

        else:
            raise self.logger.error("Task type Error!")


    def fit_model(self, train_loader):
        self.model.fit(train_loader[0], train_loader[1])


    def predict(self, test_loader):
        if self.task_type == 'Classification':
            return self.model.predict_proba(test_loader[0])
        else:
            return self.model.predict(test_loader[0])