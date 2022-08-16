import os
import json
import concurrent.futures

import torch
import numpy as np
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

from MolRep.Utils.utils import NumpyEncoder
from MolRep.Utils.logger import Logger
from MolRep.Utils.config_from_dict import Config

import multiprocessing

class HoldOutSelector:
    """
    Class implementing a sufficiently general framework to do model selection
    """

    def __init__(self, max_processes):
        self.max_processes = max_processes
        self.available_gpu_num = max_processes

        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'
        self.MODEL_PATH = 'model.pt'

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
                # vl = config_dict['VL_loss']

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
        HOLDOUT_MS_FOLDER = os.path.join(exp_path, 'HOLDOUT_MS')
        self.no_parallel = no_parallel

        if not os.path.exists(HOLDOUT_MS_FOLDER):
            os.makedirs(HOLDOUT_MS_FOLDER)

        config_id = 0

        pool = multiprocessing.Pool(self.max_processes)

        for config in model_configs:  # generate_grid(model_configs):

            # Create a separate folder for each experiment
            exp_config_name = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(config_id + 1))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)

            json_config = os.path.join(exp_config_name, self._CONFIG_FILENAME)
            model_state_dict = os.path.join(exp_config_name, self.MODEL_PATH)

            if other is None:
                other = {'model_path': model_state_dict}
            else:
                other['model_path'] = model_state_dict


            if not os.path.exists(json_config):
                if not no_parallel:
                    # pool.submit(self._model_selection_helper, dataset_getter, experiment_class, config, dataset_config,
                    #                                                exp_config_name, other) 
                    pool.apply_async(self._model_selection_helper, (dataset_getter, experiment_class, config, dataset_config,
                                                                    exp_config_name, other, ))
                else:  # No-parallel
                    self._model_selection_helper(dataset_getter, experiment_class, config, dataset_config, exp_config_name,
                                                 other)
            else:
                # Do not recompute experiments for this fold.
                print(f"Config {json_config} already present! Shutting down to prevent loss of previous experiments")
                continue

            config_id += 1

        # pool.shutdown()  
        pool.close()  # wait the batch of configs to terminate
        pool.join()

        best_config = self.process_results(HOLDOUT_MS_FOLDER, config_id)

        with open(os.path.join(HOLDOUT_MS_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(best_config, fp, cls=NumpyEncoder)

        return best_config

    def _model_selection_helper(self, dataset_getter, experiment_class, config, dataset_config, exp_config_name,
                                other=None):
        """
        :param dataset_getter:
        :param experiment_class:
        :param config:
        :param exp_config_name:
        :param other:
        :return:
        """
        # Create the experiment object which will be responsible for running a specific experiment
        experiment = experiment_class(config, dataset_config, exp_config_name)
        metric_type = dataset_config['metric_type'][0] if isinstance(dataset_config['metric_type'], list) else dataset_config['metric_type']

        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
        logger.log('Configuration: ' + str(experiment.model_config))

        config_filename = os.path.join(experiment.exp_path, self._CONFIG_FILENAME)

        # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

        selection_dict = {
            'config': experiment.model_config.config_dict,
            'task_type': dataset_config["task_type"],
            'TR_score': 0.,
            'VL_score': 0.,
            'VL_loss': 0.,
        }

        dataset_getter.set_inner_k(0)  # need to stay this way
        training_score, training_loss, validation_score, validation_loss = experiment.run_valid(dataset_getter, logger, other=other)
        
        selection_dict['TR_score'] = training_score[metric_type]
        selection_dict['VL_score'] = validation_score[metric_type]
        selection_dict['VL_loss'] = validation_loss

        logger.log('TR Accuracy: ' + str(training_score) + ' VL Accuracy: ' + str(validation_score)+ ' VL Loss: ' + str(validation_loss))

        with open(config_filename, 'w') as fp:
            json.dump(selection_dict, fp, cls=NumpyEncoder)



config_id = 1

class HoldOutBayesianSelector:
    """
    Class implementing a sufficiently general framework to do model selection
    """

    def __init__(self, max_processes):
        self.max_processes = max_processes

        # Create the experiments folder straight away
        self._CONFIG_BASE = 'config_'
        self._CONFIG_FILENAME = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'
        self._CONVERGENCE_FIGURE = 'convergence_figure.png'

        
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
        HOLDOUT_MS_FOLDER = os.path.join(exp_path, 'HOLDOUT_MS')
        if not os.path.exists(HOLDOUT_MS_FOLDER):
            os.makedirs(HOLDOUT_MS_FOLDER)

        configs_init = model_configs[0]
        x0 = list()
        search_space = list()
        definite_space = dict()
        for para_name, para_values in model_configs.config_dict.items():

            if para_name in ['model', 'device', 'optimizer', 'scheduler', 'early_stopper', 'gradient_clipping']:
                definite_space[para_name] = para_values[0]
                continue

            if len(para_values) > 1:
            # if type(para_values[0]) == int or type(para_values[0]) == str or type(para_values[0]) == bool or type(para_values[0]) == list:
                if type(para_values[0]) == int and (para_name.find('size') != -1 or para_name.find('dim') != -1):
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
                
                elif type(para_values[0]) == list:
                    x0.append(list(configs_init[para_name]))
                    search_space.append(Categorical(para_values, name=para_name, transform='identity'))
                
                elif type(para_values[0]) == float:
                    x0.append(float(configs_init[para_name]))
                    search_space.append(Real(min(para_values), max(para_values), prior='uniform', name=para_name))
                
                else:
                    definite_space[para_name] = para_values[0]

            else:
                definite_space[para_name] = para_values[0]


        @use_named_args(search_space)
        def bayesian_optimizer_func(**params):
            # print('params', params)
            params = {k: int(v) if type(v) is np.int64 else v for k,v in params.items()}
            all_params = {**params, **definite_space}
            config = Config.from_dict(all_params)

            # Create a separate folder for each experiment
            global config_id 
            exp_config_name = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(config_id))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)

            # Set up a log file for this experiment (run in a separate process)
            logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='a')
            logger.log('Configuration: ' + str(config.config_dict))
            config_filename = os.path.join(exp_config_name, self._CONFIG_FILENAME)

            selection_dict = {
                'config': config.config_dict,
                'task_type': dataset_config["task_type"],
                'TR_score': 0.,
                'VL_score': 0.,
                'VL_loss': 0.,
            }

            # Create the experiment object which will be responsible for running a specific experiment
            experiment = experiment_class(config, dataset_config, exp_config_name)
            metric_type = dataset_config['metric_type'][0] if isinstance(dataset_config['metric_type'], list) else dataset_config['metric_type']

            dataset_getter.set_inner_k(0)  # need to stay this way
            training_score, training_loss, validation_score, best_validation_score, validation_loss = experiment.run_valid(dataset_getter, logger, other=other)

            selection_dict['TR_score'] = training_score[metric_type]
            selection_dict['VL_score'] = validation_score[metric_type]
            selection_dict['VL_best_score'] = best_validation_score[metric_type]
            selection_dict['VL_loss'] = validation_loss

            logger.log(f'TR {metric_type}: ' + str(training_score[metric_type]) + f', VL {metric_type}: ' + str(validation_score[metric_type]) + f', VL best {metric_type}: ' + str(best_validation_score[metric_type]))
            # logger.log(f"TR {metric_type}: {training_score:.3f}, VL {metric_type}: {selection_dict['VL_score']:.3f}, VL Best {metric_type}: {selection_dict['VL_best_score']:.3f}")


            with open(config_filename, 'w') as fp:
                json.dump(selection_dict, fp, cls=NumpyEncoder)

            config_id += 1
            return selection_dict['VL_loss']


        # perform optimization
        optimizer_result = gp_minimize(
            bayesian_optimizer_func, 
            search_space,
            acq_func="LCB",
            n_calls=100,
            x0=x0,
            random_state=0,
            n_jobs=1
        )

        import matplotlib.pyplot as plt
        from skopt.plots import plot_convergence
        plot_convergence(optimizer_result)
        plt.savefig(os.path.join(HOLDOUT_MS_FOLDER, self._CONVERGENCE_FIGURE))

        best_config = self.process_results(HOLDOUT_MS_FOLDER, config_id)

        with open(os.path.join(HOLDOUT_MS_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(best_config, fp, cls=NumpyEncoder)

        return best_config
