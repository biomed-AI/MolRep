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
# DATASET_NAME = 'PPB'  
# DATASET_NAME = 'hERG1'  
# DATASET_NAME = 'hERG2'  
# DATASET_NAME = 'hERG3'  
# DATASET_NAME = 'Ames'  

# define the PATH of your data.
# BBB
DATASET_PATH = './ADMET_V2.0/Distribution/BBB/BBB_BBB_Classification_V1.2.csv' 
# PPB
# DATASET_PATH = './ADMET_V2.0/Distribution/PPB/PPB_PPB-GS-data_Regression_V2.0.csv'
# hERG
# DATASET_PATH = './ADMET_V2.0/Toxicity/hERG/Herg/hERG_Herg_Classification_V1.0.csv'
# DATASET_PATH = './ADMET_V2.0/Toxicity/hERG/EU-data/hERG_Herg_EU-data_1_Regression_V1.2.csv'
# DATASET_PATH = './ADMET_V2.0/Toxicity/hERG/EU-data/hERG_Herg_EU-data_2_Regression_V1.2.csv'
# Ames
# DATASET_PATH = './ADMET_V2.0/Toxicity/Ames/Ames/AMES_AMES_Classification_V1.2.csv'

# define the column name of SMILES in your data
SMILES_COLUMN = 'COMPOUND_SMILES'
# the column names of TARGET NAME in your data. Must be a List.
# TARGET_COLUMNS = ['REG_LABEL']
TARGET_COLUMNS = ['CLF_LABEL']
# define the task type. Classification or Regression
# TASK_TYPE = 'Regression'
TASK_TYPE = 'Classification'
# define the metric type, such as auc or rmse
METRIC_TYPE =[ 'acc','positive_pct.','precision','recall','F1','auc','Count']       # for classification
# METRIC_TYPE = ['rmse', 'mae','R2','pearson','spearman','Count']                     # for regression
# define the split data type, such as random, stratified, scaffold. NOTE that stratified only applies to single property
SPLIT_TYPE = 'specific'
SPLIT_COLUMN = 'SPLIT'

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
    inner_k = _FOLDS,
    config_dir = MODEL_CONFIG_DIR,
    output_dir=OUTPUT_DIR
)

config_id = 0   # the idx of model configs since there are more than 100 combinations of hyper-parameters.
KFOLD_FOLDER = os.path.join(exp_path, str(_FOLDS) + '_FOLD_MS')
exp_config_name = os.path.join(KFOLD_FOLDER, _CONFIG_BASE + str(config_id + 1))
config_filename = os.path.join(exp_config_name, _CONFIG_FILENAME)
if not os.path.exists(exp_config_name):
    os.makedirs(exp_config_name)

config = model_configurations[config_id]

# model configs could be change
# for example:
# config['device'] = 'cpu' or config['batch_size'] = 32

logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='w')
logger.log('Configuration: ' + str(config))

k_fold_dict = {
    'config': config,
    'folds': [{} for _ in range(_FOLDS)],
    'avg_TR_score': 0.,
    'avg_VL_score': 0.,
    'std_TR_score': 0.,
    'std_VL_score': 0.
}


dataset_getter = MolRep.construct_dataloader(dataset)
for k in range(1):
    logger.log(f"Training in Fold: {k+1}")
    # dataset_getter.set_inner_k(k)
    dataset_getter.set_inner_k(-1)


    fold_exp_folder = os.path.join(exp_config_name, 'FOLD_' + str(k + 1))
    # Create the experiment object which will be responsible for running a specific experiment
    experiment = EndToEndExperiment(config, dataset_config, fold_exp_folder)

    model_path = os.path.join(fold_exp_folder, f"{MODEL_NAME}_{DATASET_NAME}_fold_{k}.pt")
    training_score, test_score = experiment.run_test(dataset_getter, logger, other={'model_path': model_path})

    # print('training_score:', training_score, 'validation_score:',validation_score)
    logger.log(str(k+1) + ' split, TR Score: ' + str(training_score) +
                ' TS Score: ' + str(test_score))

#     k_fold_dict['folds'][k]['TR_score'] = training_score
#     k_fold_dict['folds'][k]['TS_score'] = test_score

# tr_scores = np.array([k_fold_dict['folds'][k]['TR_score'] for k in range(_FOLDS)])
# vl_scores = np.array([k_fold_dict['folds'][k]['TS_score'] for k in range(_FOLDS)])

# k_fold_dict['avg_TR_score'] = tr_scores.mean()
# k_fold_dict['std_TR_score'] = tr_scores.std()
# k_fold_dict['avg_VL_score'] = vl_scores.mean()
# k_fold_dict['std_VL_score'] = vl_scores.std()

# log_str = f"TR avg is %.4f std is %.4f; VL avg is %.4f std is %.4f" % (
#             k_fold_dict['avg_TR_score'], k_fold_dict['std_TR_score'], k_fold_dict['avg_VL_score'], k_fold_dict['std_VL_score']
#         )
# logger.log(log_str)