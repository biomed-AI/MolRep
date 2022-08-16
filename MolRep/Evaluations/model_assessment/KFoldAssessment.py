
import os
import json

import numpy as np
import concurrent.futures

from MolRep.Utils.logger import Logger
from MolRep.Utils.config_from_dict import Config
from MolRep.Evaluations.DataloaderWrapper import DataLoaderWrapper

from MolRep.Utils.utils import *

class KFoldAssessment:
    """
    Class implementing a sufficiently general framework to do model ASSESSMENT
    """

    def __init__(self, outer_folds, model_selector, exp_path, model_configs, dataset_config, outer_processes=2):
        self.outer_folds = outer_folds
        self.outer_processes = outer_processes
        self.model_selector = model_selector
        self.model_configs = model_configs  # Dictionary with key:list of possible values
        self.dataset_config = dataset_config

        # Create the experiments folder straight away
        assert self.outer_folds is None
            # self.outer_folds = 1
        self.exp_path = exp_path
        self.__NESTED_FOLDER = os.path.join(exp_path, str(self.outer_folds) + '_NESTED_CV')
        self.__OUTER_FOLD_BASE = 'OUTER_FOLD_'
        self._OUTER_RESULTS_FILENAME = 'outer_results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'


    def process_results(self):

        outer_TR_scores = []
        outer_TS_scores = []
        assessment_results = {}

        for i in range(1, self.outer_folds+1):
            try:
                config_filename = os.path.join(self.__NESTED_FOLDER, self.__OUTER_FOLD_BASE + str(i),
                                               self._OUTER_RESULTS_FILENAME)

                with open(config_filename, 'r') as fp:
                    outer_fold_scores = json.load(fp)

                    outer_TR_scores.append(outer_fold_scores['OUTER_TR'])
                    outer_TS_scores.append(outer_fold_scores['OUTER_TS'])

            except Exception as e:
                print("here!", config_filename)
                print(e)

        outer_TR_scores = np.array(outer_TR_scores)
        outer_TS_scores = np.array(outer_TS_scores)

        assessment_results['avg_TR_score'] = outer_TR_scores.mean()
        assessment_results['std_TR_score'] = outer_TR_scores.std()
        assessment_results['avg_TS_score'] = outer_TS_scores.mean()
        assessment_results['std_TS_score'] = outer_TS_scores.std()

        with open(os.path.join(self.__NESTED_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(assessment_results, fp)


    def risk_assessment(self, dataset, experiment_class, no_parallel=False, other=None):
        """
        :param experiment_class: the kind of experiment used
        :param no_parallel:
        :param other: anything you want to share across processes
        :return: An average over the outer test folds. RETURNS AN ESTIMATE, NOT A MODEL!!!
        """
        if not os.path.exists(self.__NESTED_FOLDER):
            os.makedirs(self.__NESTED_FOLDER)

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.outer_processes)
        for outer_k in range(self.outer_folds):

            # Create a separate folder for each experiment
            kfold_folder = os.path.join(self.__NESTED_FOLDER, self.__OUTER_FOLD_BASE + str(outer_k + 1))
            if not os.path.exists(kfold_folder):
                os.makedirs(kfold_folder)

            json_outer_results = os.path.join(kfold_folder, self._OUTER_RESULTS_FILENAME)
            if not os.path.exists(json_outer_results):
                if not no_parallel:
                    pool.submit(self._risk_assessment_helper, dataset, outer_k,
                                experiment_class, kfold_folder, no_parallel, other)
                else:  # DEBUG
                    self._risk_assessment_helper(dataset, outer_k, experiment_class, kfold_folder, no_parallel, other)
            else:
                # Do not recompute experiments for this outer fold.
                print(f"File {json_outer_results} already present! Shutting down to prevent loss of previous experiments")
                continue

            # Create a separate folder for each experiment
            # kfold_folder = os.path.join(self.__NESTED_FOLDER, self.__OUTER_FOLD_BASE + str(outer_k + 1))
            # if not os.path.exists(kfold_folder):
            #     os.makedirs(kfold_folder)
            # else:
            #     # Do not recompute experiments for this outer fold.
            #     print(f"Outer folder {outer_k} already present! Shutting down to prevent loss of previous experiments")
            #     continue
        pool.shutdown()  # wait the batch of configs to terminate

        self.process_results()


    def _risk_assessment_helper(self, dataset, outer_k, experiment_class, exp_path, no_parallel=False, other=None):

        dataset_getter = DataLoaderWrapper(dataset, outer_k)

        best_config = self.model_selector.model_selection(dataset_getter, experiment_class, exp_path,
                                                          self.model_configs, self.dataset_config, no_parallel, other)

        # Retrain with the best configuration and test
        experiment = experiment_class(best_config['config'], self.dataset_config, exp_path)
        metric_type = self.dataset_config['metric_type'][0] if isinstance(self.dataset_config['metric_type'], list) else self.dataset_config['metric_type']

        # Set up a log file for this experiment (run in a separate process)

        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        dataset_getter.set_inner_k(None)  # needs to stay None

        training_scores, test_scores = [], []
        # Mitigate bad random initializations
        for i in range(5):
            logger.log(f"\nTraining Run {i + 1}")
            model_path = str(os.path.join(experiment.exp_path, f'experiment_run_{i}.pt'))
            training_score, _, test_score = experiment.run_test(dataset_getter, logger, other={'model_path': model_path})
            logger.log(f'Final training run {i + 1}: train {metric_type} {training_score[metric_type]}, test {metric_type} {test_score[metric_type]}')

            training_scores.append(training_score[metric_type])
            test_scores.append(test_score[metric_type])

        training_score = np.mean(training_scores)
        training_score_std = np.std(training_scores)
        test_score = np.mean(test_scores)
        test_score_std = np.std(test_scores)

        logger.log('End of Outer fold. TR score: [{:.04f}] ({:.5f}), TS score: [{:.04f}] ({:.5f})'.format(training_score, training_score_std, test_score, test_score_std))

        with open(os.path.join(exp_path, self._OUTER_RESULTS_FILENAME), 'w') as fp:
            json.dump({'best_config': best_config, 'OUTER_TR': training_score, 'OUTER_TR_STD':training_score_std, 'OUTER_TS': test_score, 'OUTER_TS_STD': test_score_std}, fp)
