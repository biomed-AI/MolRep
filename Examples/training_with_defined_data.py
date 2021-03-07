#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
sys.path.append('../')
import numpy as np

from MolRep import MolRep
from MolRep.Utils.logger import Logger
from MolRep.Utils.config_from_dict import Config
from MolRep.Experiments.experiments import EndToEndExperiment


# In[2]:


MODEL_CONFIG_DIR = '../MolRep/Configs' # Need to set! The directory of Model Configurations files, such as config_CMPNN.yml.
DATASET_DIR = '../MolRep/Datasets'     # Need to set! The directory of Datasets downloaded from Google Drive.
OUTPUT_DIR = '../Outputs/'

# Output file name
_CONFIG_BASE = 'config_'
_CONFIG_FILENAME = 'config_results.json'

# Args
_FOLDS = 5
MODEL_NAME = 'MorganFP' #'MolecularFingerprint' #'CMPNN'
DATASET_NAME = 'BBBP'


# In[3]:


dataset_config, dataset, model_configurations, model_selector, exp_path = MolRep.construct_dataset(
        dataset_name = DATASET_NAME,
        model_name = MODEL_NAME,
        inner_k = _FOLDS,
        config_dir = MODEL_CONFIG_DIR,
        datasets_dir = DATASET_DIR,
        output_dir=OUTPUT_DIR
)


# In[ ]:





# In[4]:


config_id = 0  # the idx of model config since there are more than 100 combinations of hyper-parameters.
KFOLD_FOLDER = os.path.join(exp_path, str(_FOLDS) + '_FOLD_MS')
exp_config_name = os.path.join(KFOLD_FOLDER, _CONFIG_BASE + str(config_id + 1))
config_filename = os.path.join(exp_config_name, _CONFIG_FILENAME)
if not os.path.exists(exp_config_name):
    os.makedirs(exp_config_name)


# In[ ]:





# In[5]:


config = model_configurations[config_id]

# model configs could be change
# for example:
# config['device'] = 'cpu' or config['batch_size'] = 32

logger = Logger(str(os.path.join(exp_config_name, 'experiment.log')), mode='w')
logger.log('Configuration: ' + str(config))


# In[ ]:





# In[6]:


k_fold_dict = {
    'config': config,
    'folds': [{} for _ in range(_FOLDS)],
    'avg_TR_score': 0.,
    'avg_VL_score': 0.,
    'std_TR_score': 0.,
    'std_VL_score': 0.
}


# In[ ]:


dataset_getter = MolRep.construct_dataloader(dataset)
for k in range(_FOLDS):
    logger.log(f"Training in Fold: {k+1}")
    dataset_getter.set_inner_k(k)

    fold_exp_folder = os.path.join(exp_config_name, 'FOLD_' + str(k + 1))
    # Create the experiment object which will be responsible for running a specific experiment
    experiment = EndToEndExperiment(config, dataset_config, fold_exp_folder)

    model_path = os.path.join(fold_exp_folder, f"{MODEL_NAME}_{DATASET_NAME}_fold_{k}.pt")
    training_score, validation_score, validation_loss = experiment.run_valid(dataset_getter, logger, other={'model_path': model_path})

    # print('training_score:', training_score, 'validation_score:',validation_score)
    logger.log(str(k+1) + ' split, TR Score: ' + str(training_score) +
                ' VL Score: ' + str(validation_score))

    k_fold_dict['folds'][k]['TR_score'] = training_score
    k_fold_dict['folds'][k]['VL_score'] = validation_score

tr_scores = np.array([k_fold_dict['folds'][k]['TR_score'] for k in range(_FOLDS)])
vl_scores = np.array([k_fold_dict['folds'][k]['VL_score'] for k in range(_FOLDS)])

k_fold_dict['avg_TR_score'] = tr_scores.mean()
k_fold_dict['std_TR_score'] = tr_scores.std()
k_fold_dict['avg_VL_score'] = vl_scores.mean()
k_fold_dict['std_VL_score'] = vl_scores.std()


log_str = f"TR avg is %.4f std is %.4f; VL avg is %.4f std is %.4f" % (
            k_fold_dict['avg_TR_score'], k_fold_dict['std_TR_score'], k_fold_dict['avg_VL_score'], k_fold_dict['std_VL_score']
        )
logger.log(log_str)


# In[ ]:




