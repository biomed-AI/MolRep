
import os
import json
import pdb
import time, datetime

import numpy as np
import concurrent.futures

from MolRep.Utils.config_from_dict import Config
from MolRep.Evaluations.DataloaderWrapper import DataLoaderWrapper

from MolRep.Utils.utils import *
from MolRep.Utils.logger import Logger

class HoldOutAssessment:
    """
    Class implementing a sufficiently general framework to do model ASSESSMENT
    """

    def __init__(self, model_selector, exp_path, model_configs, dataset_config, max_processes=2):

        self.max_processes = max_processes
        self.model_selector = model_selector
        self.model_configs = model_configs  # Dictionary with key:list of possible values
        self.dataset_config = dataset_config

        # Create the experiments folder straight away
        self.exp_path = exp_path
        self._HOLDOUT_FOLDER = os.path.join(exp_path, 'HOLDOUT_ASS')
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def risk_assessment(self, dataset, experiment_class, no_parallel=False, other=None):
        """
        :param experiment_class: the kind of experiment used
        :param no_parallel:
        :return: An average over the outer test folds. RETURNS AN ESTIMATE, NOT A MODEL!!!
        """
        if not os.path.exists(self._HOLDOUT_FOLDER):
            os.makedirs(self._HOLDOUT_FOLDER)

        self._risk_assessment_helper(dataset, experiment_class, self._HOLDOUT_FOLDER, no_parallel, other)


    def _risk_assessment_helper(self, dataset, experiment_class, exp_path, no_parallel=False, other=None):


        start_time = time.time()

        dataset_getter = DataLoaderWrapper(dataset)
        best_config = self.model_selector.model_selection(dataset_getter, experiment_class, exp_path,
                                                          self.model_configs, self.dataset_config, no_parallel, other)

        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Model Selection time {}'.format(total_time_str)) 


        # Retrain with the best configuration and test
        experiment = experiment_class(best_config['config'], self.dataset_config, exp_path)
        metric_type = self.dataset_config['metric_type'][0] if isinstance(self.dataset_config['metric_type'], list) else self.dataset_config['metric_type']

        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        dataset_getter.set_inner_k(None)

        training_scores, test_scores = [], []
        # Mitigate bad random initializations
        for i in range(5):
            logger.log(f"\nTraining Run {i+1}")
            model_path = str(os.path.join(experiment.exp_path, f'experiment_run_{i + 1}.pt'))
            training_score, _, test_score = experiment.run_test(dataset_getter, logger, other={'model_path': model_path})
            logger.log(f'Final training run {i + 1}: train {metric_type} {training_score[metric_type]}, test {metric_type} {test_score[metric_type]}')

            training_scores.append(training_score[metric_type])
            test_scores.append(test_score[metric_type])

        training_score = np.mean(training_scores)
        training_score_std = np.std(training_scores)
        test_score = np.mean(test_scores)
        test_score_std = np.std(test_scores)


        # logger.log('TR score: ' + str(training_score) + ' TS score: ' + str(test_score))
        logger.log(f"\nTR {metric_type}: " + "[{:.04f}] ({:.5f});".format(training_score, training_score_std) + f" TS {metric_type}:" + " [{:.04f}] ({:.5f})".format(test_score, test_score_std))

        with open(os.path.join(self._HOLDOUT_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump({'best_config': best_config, 'HOLDOUT_TR': training_score, 'HOLDOUT_TR_STD': training_score_std, 'HOLDOUT_TS': test_score, 'HOLDOUT_TS_STD':test_score_std}, fp)