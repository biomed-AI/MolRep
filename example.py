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
# MODEL_NAME = 'MAT'

# define your dataset name
# DATASET_NAME = 'BBB'  
# DATASET_NAME = 'PPB'  
# DATASET_NAME = 'hERG1'  
# DATASET_NAME = 'hERG2'  
# DATASET_NAME = 'hERG3'  
DATASET_NAME = 'Ames'  

# define the PATH of your data.
# BBB
# DATASET_PATH = './ADMET_V2.0/Distribution/BBB/BBB_BBB_Classification_V1.2.csv' 
# PPB
# DATASET_PATH = './ADMET_V2.0/Distribution/PPB/PPB_PPB-GS-data_Regression_V2.0.csv'
# hERG
# DATASET_PATH = './ADMET_V2.0/Toxicity/hERG/Herg/hERG_Herg_Classification_V1.0.csv'
# DATASET_PATH = './ADMET_V2.0/Toxicity/hERG/EU-data/hERG_Herg_EU-data_1_Regression_V1.2.csv'
# DATASET_PATH = './ADMET_V2.0/Toxicity/hERG/EU-data/hERG_Herg_EU-data_2_Regression_V1.2.csv'
# Ames
DATASET_PATH = './ADMET_V2.0/Toxicity/Ames/Ames/AMES_AMES_Classification_V1.2.csv'
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
# SPLIT_TYPE = 'random'

# If you have split your data to train/test, you could define SPLIT_TYPE = 'specific' and save your split in CSV file.
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
    config_dir = MODEL_CONFIG_DIR,
    output_dir=OUTPUT_DIR
)

model_assesser = MolRep.construct_assesser(model_selector, exp_path, model_configurations, dataset_config
                                          )

model_assesser.risk_assessment(dataset, EndToEndExperiment, debug=True)