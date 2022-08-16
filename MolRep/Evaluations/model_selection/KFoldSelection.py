import os
import json

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

import numpy as np
import concurrent.futures
from copy import deepcopy

from MolRep.Utils.logger import Logger
from MolRep.Utils.config_from_dict import Config

import multiprocessing

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



class KFoldSelector:
    """
    Class implementing a sufficiently general framework to do model selection
    """

    def __init__(self, folds, max_processes):
        self.folds = folds
        self.max_processes = max_processes

        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'
        self.MODEL_PATH = 'model.pt'

    def process_results(self, KFOLD_FOLDER, no_configurations):

        best_avg_vl = 0.
        best_std_vl = 100.

        for i in range(1, no_configurations+1):
            try:
                config_filename = os.path.join(KFOLD_FOLDER, self._CONFIG_BASE + str(i), self._CONFIG_FILENAME)

                with open(config_filename, 'r') as fp:
                    config_dict = json.load(fp)

                avg_vl = config_dict['avg_VL_score']
                std_vl = config_dict['std_VL_score']

                if (best_avg_vl < avg_vl) or (best_avg_vl == avg_vl and best_std_vl > std_vl):
                    best_i = i
                    best_avg_vl = avg_vl
                    best_config = config_dict

            except Exception as e:
                print(e)

        print('Model selection winner for experiment', KFOLD_FOLDER, 'is config ', best_i, ':')
        for k in best_config.keys():
            print('\t', k, ':', best_config[k])

        return best_config

    def model_selection(self, dataset_getter, experiment_class, exp_path, model_configs, dataset_config, no_parallel=False, other=None):
        """
        :param experiment_class: the kind of experiment used
        :param no_parallel:
        :return: the best performing configuration on average over the k folds. TL;DR RETURNS A MODEL, NOT AN ESTIMATE!
        """

        exp_path = exp_path
        KFOLD_FOLDER = os.path.join(exp_path, str(self.folds) + '_FOLD_MS')

        if not os.path.exists(KFOLD_FOLDER):
            os.makedirs(KFOLD_FOLDER)

        config_id = 0
        pool = multiprocessing.Pool(self.max_processes)
        for config in model_configs:

            # I need to make a copy of this dictionary
            # It seems it gets shared between processes!
            cfg = deepcopy(config)

            # Create a separate folder for each experiment
            exp_config_name = os.path.join(KFOLD_FOLDER, self._CONFIG_BASE + str(config_id + 1))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)

            if not no_parallel:
                # pool.submit(self._model_selection_helper, dataset_getter, experiment_class, cfg, dataset_config,
                #             exp_config_name, other)
                pool.apply_async(self._model_selection_helper, (dataset_getter, experiment_class, config, dataset_config,
                                                                   exp_config_name, other, ))
            else:  # No-Parallel
                self._model_selection_helper(dataset_getter, experiment_class, cfg, dataset_config,
                                             exp_config_name, other)

            config_id += 1

        # pool.shutdown()
        pool.close()
        pool.join()
        best_config = self.process_results(KFOLD_FOLDER, config_id)

        with open(os.path.join(KFOLD_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(best_config, fp, cls=NumpyEncoder)

        return best_config

    def _model_selection_helper(self, dataset_getter, experiment_class, config, dataset_config, exp_config_name,
                                other=None):

        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='a')

        logger.log('Configuration: ' + str(config))

        config_filename = os.path.join(exp_config_name, self._CONFIG_FILENAME)

        # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

        k_fold_dict = {
            'config': config,
            'task_type': dataset_config["task_type"],
            'folds': [{} for _ in range(self.folds)],
            'avg_TR_score': 0.,
            'avg_VL_score': 0.,
            'avg_VL_loss': 0.,
            'std_TR_score': 0.,
            'std_VL_score': 0.,
            'std_VL_loss': 0.,
        }

        for k in range(self.folds):

            dataset_getter.set_inner_k(k)

            fold_exp_folder = os.path.join(exp_config_name, 'FOLD_' + str(k + 1))
            model_state_dict = os.path.join(fold_exp_folder, self.MODEL_PATH)

            if other is None:
                other = {'model_path': model_state_dict}
            else:
                other['model_path'] = model_state_dict

            # Create the experiment object which will be responsible for running a specific experiment
            experiment = experiment_class(config, dataset_config, fold_exp_folder)
            metric_type = dataset_config['metric_type'][0] if isinstance(dataset_config['metric_type'], list) else dataset_config['metric_type']

            training_score, training_loss, validation_score, best_validation_score, validation_loss = experiment.run_valid(dataset_getter, logger, other=other)

            print('training_score:', training_score, 'validation_score:',validation_score)
            logger.log(str(k+1) + f' split, TR {metric_type}: ' + str(training_score) +
                       f', VL {metric_type}: ' + str(validation_score) + f', VL best {metric_type}: ' + str(best_validation_score))

            k_fold_dict['folds'][k]['TR_score'] = training_score[metric_type]
            k_fold_dict['folds'][k]['VL_score'] = validation_score[metric_type]
            k_fold_dict['folds'][k]['VL_loss'] = validation_loss

        tr_scores = np.array([k_fold_dict['folds'][k]['TR_score'] for k in range(self.folds)])
        vl_scores = np.array([k_fold_dict['folds'][k]['VL_score'] for k in range(self.folds)])
        vl_losses = np.array([k_fold_dict['folds'][k]['VL_loss'] for k in range(self.folds)])

        k_fold_dict['avg_TR_score'] = tr_scores.mean()
        k_fold_dict['std_TR_score'] = tr_scores.std()
        k_fold_dict['avg_VL_score'] = vl_scores.mean()
        k_fold_dict['std_VL_score'] = vl_scores.std()
        k_fold_dict['avg_VL_loss'] = vl_losses.mean()
        k_fold_dict['std_VL_loss'] = vl_losses.std()

        log_str = f"TR avg is %.4f std is %.4f; VL avg is %.4f std is %.4f" % (
            k_fold_dict['avg_TR_score'], k_fold_dict['std_TR_score'], k_fold_dict['avg_VL_score'], k_fold_dict['std_VL_score']
        )
        logger.log(log_str)

        with open(config_filename, 'w') as fp:
            print('write to {}'.format(config_filename))
            json.dump(k_fold_dict, fp, cls=NumpyEncoder)



config_id = 1

class KFoldBayesianSelector:
    """
    Class implementing a sufficiently general framework to do model selection
    """

    def __init__(self, folds, max_processes):
        self.folds = folds
        self.max_processes = max_processes

        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'

    def process_results(self, HOLDOUT_MS_FOLDER, no_configurations):

        best_vl = 0.
        best_i = -1
        best_config = {}

        for i in range(1, no_configurations+1):
            try:
                config_filename = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(i),
                                               self._CONFIG_FILENAME)

                with open(config_filename, 'r') as fp:
                    config_dict = json.load(fp)

                vl = config_dict['VL_score']
                if vl >= best_vl:
                    best_i = i
                    best_vl = vl
                    best_config = config_dict

            except Exception as e:
                print(e)

        print('Model selection winner for experiment', HOLDOUT_MS_FOLDER, 'is config ', best_i, ':')
        for k in best_config.keys():
            print('\t', k, ':', best_config[k])

        return best_config


    def model_selection(self, dataset_getter, experiment_class, exp_path, model_configs, dataset_config, no_parallel=False, other=None):
        """
        :param experiment_class: the kind of experiment used
        :param no_parallel:
        :return: the best performing configuration on average over the k folds. TL;DR RETURNS A MODEL, NOT AN ESTIMATE!
        """

        exp_path = exp_path
        KFOLD_FOLDER = os.path.join(exp_path, str(self.folds) + '_FOLD_MS')
        if not os.path.exists(KFOLD_FOLDER):
            os.makedirs(KFOLD_FOLDER)

        configs_init = model_configs[0].config_dict
        x0 = list()
        search_space = list()
        definite_space = dict()
        for para_name, para_values in model_configs.config_dict.items():

            if para_name in ['model', 'device', 'optimizer', 'scheduler', 'early_stopper', 'gradient_clipping']:
                definite_space[para_name] = para_values[0]
                continue

            if len(para_values) > 1:
            # if type(para_values[0]) == int or type(para_values[0]) == str or type(para_values[0]) == bool or type(para_values[0]) == list:
                if type(para_values[0]) == int and para_name.find('size') != -1:
                    x0.append(int(configs_init[para_name]))
                    search_space.append(Categorical(para_values, name=para_name, transform='identity'))
                
                elif type(para_values[0]) == int:
                    x0.append(int(configs_init[para_name]))
                    search_space.append(Integer(min(para_values), max(para_values), name=para_name, dtype='int'))
                
                elif type(para_values[0]) == str:
                    x0.append(str(configs_init[para_name]))
                    search_space.append(Categorical(para_values, name=para_name, transform='identity'))
                
                elif type(para_values[0]) == bool:
                    x0.append(bool(configs_init[para_name]))
                    search_space.append(Categorical(para_values, name=para_name, transform='identity'))
                
                # elif type(para_values[0]) == list:
                #     x0.append(list(configs_init[para_name]))
                #     search_space.append(Categorical(para_values, name=para_name))
                
                elif type(para_values[0]) == float:
                    x0.append(float(configs_init[para_name]))
                    search_space.append(Real(min(para_values), max(para_values), prior='uniform', name=para_name))
                
                else:
                    definite_space[para_name] = para_values[0]

            else:
                definite_space[para_name] = para_values[0]


        @use_named_args(search_space)
        def bayesian_optimizer_func(**params):
            # print(params)
            all_params = {**params, **definite_space}
            config = Config.from_dict(all_params)

            global config_id
            # Create a separate folder for each experiment
            exp_config_name = exp_config_name = os.path.join(KFOLD_FOLDER, self._CONFIG_BASE + str(config_id + 1))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)
            # Set up a log file for this experiment (run in a separate process)
            logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='a')
            logger.log('Configuration: ' + str(config))
            config_filename = os.path.join(exp_config_name, self._CONFIG_FILENAME)

            # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

            k_fold_dict = {
                'config': config.config_dict,
                'task_type': dataset_config["task_type"],
                'folds': [{} for _ in range(self.folds)],
                'avg_TR_score': 0.,
                'avg_VL_score': 0.,
                'avg_VL_loss': 0.,
                'std_TR_score': 0.,
                'std_VL_score': 0.,
                'std_VL_loss': 0.,
            }

            metric_type = dataset_config['metric_type'][0] if isinstance(dataset_config['metric_type'], list) else dataset_config['metric_type']

            for k in range(self.folds):
                dataset_getter.set_inner_k(k)
                fold_exp_folder = os.path.join(exp_config_name, 'FOLD_' + str(k + 1))
                # Create the experiment object which will be responsible for running a specific experiment
                experiment = experiment_class(config, dataset_config, fold_exp_folder)

                training_score, training_loss, validation_score, best_validation_score, validation_loss = experiment.run_valid(dataset_getter, logger, other=other)

                # print('training_score:', training_score, 'validation_score:',validation_score)
                logger.log(str(k+1) + f' split, TR {metric_type}: ' + str(training_score) +
                        f', VL {metric_type}: ' + str(validation_score) + f', VL best {metric_type}: ' + str(best_validation_score))

            tr_scores = np.array([k_fold_dict['folds'][k]['TR_score'] for k in range(self.folds)])
            vl_scores = np.array([k_fold_dict['folds'][k]['VL_score'] for k in range(self.folds)])
            vl_losses = np.array([k_fold_dict['folds'][k]['VL_loss'] for k in range(self.folds)])

            k_fold_dict['avg_TR_score'] = tr_scores.mean()
            k_fold_dict['std_TR_score'] = tr_scores.std()
            k_fold_dict['avg_VL_score'] = vl_scores.mean()
            k_fold_dict['std_VL_score'] = vl_scores.std()
            k_fold_dict['avg_VL_loss'] = vl_losses.mean()
            k_fold_dict['std_VL_loss'] = vl_losses.std()

            log_str = f"TR avg is %.4f std is %.4f; VL avg is %.4f std is %.4f" % (
                k_fold_dict['avg_TR_score'], k_fold_dict['std_TR_score'], k_fold_dict['avg_VL_score'], k_fold_dict['std_VL_score']
            )
            logger.log(log_str)

            with open(config_filename, 'w') as fp:
                print('write to {}'.format(config_filename))
                json.dump(k_fold_dict, fp, cls=NumpyEncoder)

            return k_fold_dict['avg_VL_loss']

        # perform optimization
        optimizer_result = gp_minimize(
            bayesian_optimizer_func, 
            search_space,
            acq_func="LCB",
            x0=x0
        )

        # best_config = dict()
        # for para, best_val in optimizer_result.items():
        #     best_config[para] = best_val
        # for para, def_val in definite_space.items():
        #     best_config[para] = def_val


        best_config = self.process_results(KFOLD_FOLDER, config_id)
        with open(os.path.join(KFOLD_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(best_config, fp, cls=NumpyEncoder)

        return best_config