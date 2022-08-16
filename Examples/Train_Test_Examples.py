'''
CUDA_VISIBLE_DEVICES=2 python Train_Test_Examples.py --model_name GraphSAGE \
                                                     --data_path ../dataset/BBBP/BBBP.csv \
                                                     --dataset_name BBBP \
                                                     --smiles_col smiles \
                                                     --target_col p_np \
                                                     --task_type Classification \
                                                     --validation_size 0.1 \
                                                     --test_size 0.1 \
                                                     --output_dir ../Outputs


'''


import argparse
import os

import torch
from pathlib import Path

from MolRep.Utils.logger import Logger
from MolRep.Utils.config_from_dict import Config, Grid, DatasetConfig

from MolRep.Evaluations.DatasetWrapper import DatasetWrapper
from MolRep.Evaluations.DataloaderWrapper import DataLoaderWrapper
from MolRep.Experiments.experiments import EndToEndExperiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate structural and global importances for a molecule using graph neural-network approach."
    )

    parser.add_argument("--model_name", dest="model_name", type=str, required=True, help="Name of the GNN model.")
    parser.add_argument("--data_path", dest="data_path", type=str, required=True, help="SMILES string or path to a valid .smi file with several SMILES separated by newlines",)

    parser.add_argument("--dataset_name", dest="dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--smiles_col", dest="smiles_col", type=str, required=True, help="Name of the column with target smiles.")
    parser.add_argument("--target_col", dest="target_col", type=str, required=True, help="Name of the column with the target values",)
    parser.add_argument("--split_type", dest="split_type", type=str, required=False, default="scaffold", help="Type of Split Dataset",)
    
    parser.add_argument('--test_size', dest='test_size', type=float, default=0.)
    parser.add_argument('--validation_size', dest='validation_size', type=float, default=0.)

    parser.add_argument("--task_type", dest="task_type", type=str, required=False, default="Regression", help="Type of training tasks. Options: Regression",)
    parser.add_argument("--multiclass_num_classes", dest="multiclass_num_classes", type=int, required=False, default=1, help='multiclass num classes')
    parser.add_argument("--feature_scale", dest="feature_scale", type=bool, required=False, default=False, help="Scales the gradients by the original features.",)
    parser.add_argument("--testing", dest="testing", type=bool, required=False, default=True, help="Whether to explainer the testing set or the full dataset")
    parser.add_argument("--training",  dest="training", type=bool, required=False, default=False, help="Whether to explainer the training set or the test set")
    parser.add_argument("--output_dir", dest="output_dir", type=str, required=True, help="Output path where to store results",)

    args = parser.parse_args()
    LOGGER_BASE = os.path.join(args.output_dir, "Logger", f"{args.dataset_name}_training")
    logger = Logger(str(os.path.join(LOGGER_BASE, f"{args.model_name}_{args.dataset_name}.log")), mode='a')

    data_dir = Path('../processed_data')
    split_dir = Path('../splits')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    output_dir = Path(args.output_dir)
    args.model_path = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_name}_training", f"{args.model_name}.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.set_num_threads(1)

    data_stats = {
                'name': args.dataset_name,
                'path': args.data_path,
                'smiles_column': args.smiles_col,
                'target_columns': [args.target_col],
                'task_type': args.task_type,
                'multiclass_num_classes': args.multiclass_num_classes,
                'metric_type': 'rmse' if args.task_type == 'Regression' else ['acc', 'auc', 'f1', 'precision', 'recall'],
                'split_type': args.split_type
    }

    if args.split_type == 'defined':
        data_stats['additional_info'] = {"splits":'SPLIT'}

    config_file = '../MolRep/Configs/config_{}.yml'.format(args.model_name)
    model_configurations = Grid(config_file)
    model_configuration = Config(**model_configurations[0])
    dataset_configuration = DatasetConfig(args.dataset_name, data_dict=data_stats)

    exp_path = os.path.join(output_dir, f'{model_configuration.exp_name}_{dataset_configuration.exp_name}_training')

    dataset = DatasetWrapper(dataset_config=dataset_configuration,
                             model_name=model_configuration.exp_name,
                             split_dir=split_dir, features_dir=data_dir,
                             validation_size=args.validation_size, test_size=args.test_size,
                             outer_k=None, inner_k=None
                             )

    dataset_getter = DataLoaderWrapper(dataset, inner_k=0)

    experiment = EndToEndExperiment(model_configuration, dataset_configuration, exp_path)
    
    if not os.path.exists(args.model_path):
         train_metric, train_loss, val_metric, best_val_metric, val_loss = experiment.run_valid(dataset_getter, logger=logger, other={'model_path':args.model_path})

    _, _, test_metric = experiment.run_independent_test(dataset_getter, logger=logger, other={'model_path':args.model_path})
    logger.log('Test results: %s' % str(test_metric))