
import os
import numpy as np

from MolRep import MolRep
from MolRep.Utils.logger import Logger
from MolRep.Utils.config_from_dict import Config
from MolRep.Experiments.experiments import EndToEndExperiment

MODEL_CONFIG_DIR = './MolRep/Configs' # Need to set! The directory of Model Configurations files, such as config_CMPNN.yml.
OUTPUT_DIR = './Outputs/'


_CONFIG_BASE = 'config_'
_CONFIG_FILENAME = 'config_results.json'

_FOLDS = 5
MODEL_NAME = 'CMPNN'

# define your dataset name
DATASET_NAME = 'BBB'  
# define the PATH of your data.
DATASET_PATH = './MolRep/Datasets/BBBP/BBB.csv'
# define the column name of SMILES in your data
SMILES_COLUMN = 'smiles'
# the column names of TARGET NAME in your data. Must be a List.
TARGET_COLUMNS = ['p_np']
# define the task type. Classification or Regression
TASK_TYPE = 'Classification'
# define the metric type, such as auc or rmse
METRIC_TYPE = ['auc', 'recall']
# define the split data type, such as random, stratified, scaffold. NOTE that stratified only applies to single property
# SPLIT_TYPE = 'random'

# If you have split your data to train/test, you could define SPLIT_TYPE = 'specific' and save your split in CSV file.
SPLIT_TYPE = 'specific'
SPLIT_COLUMN = 'splits'

dataset_config, dataset, model_configurations, model_selector, exp_path = MolRep.construct_dataset(
    dataset_name = DATASET_NAME,
    model_name = MODEL_NAME,
    dataset_path = DATASET_PATH,
    smiles_column = SMILES_COLUMN,
    target_columns = TARGET_COLUMNS,
    task_type = TASK_TYPE,
    metric_type = METRIC_TYPE,
    split_type = SPLIT_TYPE,
    additional_info = {'splits': SPLIT_COLUMN},
    config_dir = MODEL_CONFIG_DIR,
    output_dir=OUTPUT_DIR
)

model_assesser = MolRep.construct_assesser(model_selector, exp_path, model_configurations, dataset_config
                                          )

model_assesser.risk_assessment(dataset, EndToEndExperiment, debug=True)

# config_id = 0   # the idx of model configs since there are more than 100 combinations of hyper-parameters.
# KFOLD_FOLDER = os.path.join(exp_path, str(_FOLDS) + '_FOLD_MS')
# exp_config_name = os.path.join(KFOLD_FOLDER, _CONFIG_BASE + str(config_id + 1))
# config_filename = os.path.join(exp_config_name, _CONFIG_FILENAME)
# if not os.path.exists(exp_config_name):
#     os.makedirs(exp_config_name)

# config = model_configurations[config_id]

# # model configs could be change
# # for example:
# # config['device'] = 'cpu' or config['batch_size'] = 32

# logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='a')
# logger.log('Configuration: ' + str(config))

# result_dict = {
#     'config': config,
#     'folds': [{} for _ in range(_FOLDS)],
#     'TR_score': 0.,
#     'VL_score': 0.,
# }

# dataset_getter = MolRep.construct_dataloader(dataset)
# dataset_getter.set_inner_k(None)

# fold_exp_folder = os.path.join(exp_config_name)
# # Create the experiment object which will be responsible for running a specific experiment
# experiment = EndToEndExperiment(config, dataset_config, fold_exp_folder)

# model_path = os.path.join(fold_exp_folder, f"{MODEL_NAME}_{DATASET_NAME}.pt")
# training_score, validation_score = experiment.run_valid(dataset_getter, logger, other={'model_path': model_path})

# # print('training_score:', training_score, 'validation_score:',validation_score)
# logger.log('TR Score: ' + str(training_score) +
#             ' VL Score: ' + str(validation_score))

# result_dict['TR_score'] = training_score
# result_dict['VL_score'] = validation_score

# tr_scores = np.array([result_dict['TR_score'] for k in range(_FOLDS)])
# vl_scores = np.array([result_dict['VL_score'] for k in range(_FOLDS)])


# log_str = f"TR is %.4f; VL is %.4f" % (
#             result_dict['TR_score'], result_dict['VL_score']
#         )
# logger.log(log_str)