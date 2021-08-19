

'''

CUDA_VISIBLE_DEVICES=2 python Explainer_Experiments.py --model_name CMPNN \
                                                       --attribution_name GradInput \
                                                       --data_path ../MolRep/Datasets/Metabolism/admet_exp_hlm_t1-2_20210412_TriCLF.csv \
                                                       --dataset_name HLM \
                                                       --smiles_col COMPOUND_SMILES \
                                                       --target_col CLF_LABEL \
                                                       --task_type Multi-Classification \
                                                       --multiclass_num_classes 3 \
                                                       --output_dir ../Outputs \
                                                       

CUDA_VISIBLE_DEVICES=2 python Explainer_Experiments.py --model_name CMPNN \
                                                       --attribution_name GradInput \
                                                       --data_path ../MolRep/Datasets/Metabolism/admet2.1_rlm_merge.csv \
                                                       --dataset_name RLM \
                                                       --smiles_col COMPOUND_SMILES \
                                                       --target_col CLF_LABEL \
                                                       --task_type Multi-Classification \
                                                       --multiclass_num_classes 3 \
                                                       --output_dir ../Outputs
'''

import argparse
import os

import torch
from pathlib import Path

from MolRep.Explainer.explainerDataWrapper import ExplainerDatasetWrapper
from MolRep.Explainer.explainerExperiments import ExplainerExperiments

from MolRep.Utils.logger import Logger

from MolRep.Utils.config_from_dict import Config, Grid, DatasetConfig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate structural and global importances for a molecule using graph neural-network approach."
    )

    parser.add_argument(
        "--model_name",
        dest="model_name",
        type=str,
        required=True,
        help="Name of the GNN model."
    )

    parser.add_argument(
        "--attribution_name",
        dest="attribution_name",
        type=str,
        required=True,
        help="Name of the Attribution model."
    )

    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        required=True,
        help="SMILES string or path to a valid .smi file with several SMILES separated by newlines",
    )

    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        type=str,
        required=True,
        help="Name of the dataset."
    )

    parser.add_argument(
        "--smiles_col",
        dest="smiles_col",
        type=str,
        required=True,
        help="Name of the column with target smiles."
    )

    parser.add_argument(
        "--target_col",
        dest="target_col",
        type=str,
        required=True,
        help="Name of the column with the target values",
    )

    parser.add_argument(
        "--attribution_path",
        dest="attribution_path",
        type=str,
        required=False,
        default=None,
        help="Path of the Attribution values",
    )

    parser.add_argument(
        "--task_type",
        dest="task_type",
        type=str,
        required=False,
        default="Regression",
        help="Type of training tasks. Options: Regression",
    )

    parser.add_argument(
        "--multiclass_num_classes",
        dest="multiclass_num_classes",
        type=int,
        required=False,
        default=1,
        help='multiclass num classes'
    )

    parser.add_argument(
        "--n_steps",
        dest="n_steps",
        type=int,
        required=False,
        default=50,
        help="Number of steps used in the Riemann approximation of the integral. Defaults to 50.",
    )

    parser.add_argument(
        "--eps",
        dest="eps",
        type=float,
        required=False,
        default=0.0001,
        help="Minimum gradient value to show. Defaults to 1e-4.",
    )

    parser.add_argument(
        "--feature_scale",
        dest="feature_scale",
        type=bool,
        required=False,
        default=False,
        help="Scales the gradients by the original features.",
    )

    parser.add_argument(
        "--add_hs",
        dest="add_hs",
        type=bool,
        required=False,
        default=False,
        help="Whether to add hydrogens to the provided molecules",
    )

    parser.add_argument(
        "--testing",
        dest="testing",
        type=bool,
        required=False,
        default=True,
        help="Whether to explainer the testing set or the full dataset"
    )

    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Output path where to store results",
    )

    args = parser.parse_args()
    LOGGER_BASE = os.path.join(args.output_dir, "Logger", f"{args.dataset_name}_explainer")
    logger = Logger(str(os.path.join(LOGGER_BASE, f"{args.model_name}_{args.dataset_name}_explainer_by_{args.attribution_name}.log")), mode='a')

    data_dir = Path('../MolRep/Data')
    split_dir = Path('../MolRep/Splits')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    svg_dir = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_name}_explainer", "SVG", f"{args.attribution_name}")

    output_dir = Path(args.output_dir)
    args.model_path = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_name}_explainer", f"{args.model_name}.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    torch.set_num_threads(1)

    data_stats = {
                'name': args.dataset_name,
                'path': args.data_path,
                'smiles_column': args.smiles_col,
                'target_columns': [args.target_col],
                'attribution_path': args.attribution_path,
                'task_type': args.task_type,
                'multiclass_num_classes': args.multiclass_num_classes,
                'metric_type': 'rmse' if args.task_type == 'Regression' else ['acc', 'auc', 'f1', 'precision', 'recall'],
                'split_type': 'defined'
    }

    if args.testing:
        data_stats['additional_info'] = {"splits":'SPLIT'}

    config_file = '../MolRep/Configs/config_{}.yml'.format(args.model_name)
    model_configurations = Grid(config_file)
    model_configuration = Config(**model_configurations[0])
    dataset_configuration = DatasetConfig(args.dataset_name, data_dict=data_stats)

    exp_path = os.path.join(output_dir, f'{model_configuration.exp_name}_{dataset_configuration.exp_name}_explainer')

    dataset = ExplainerDatasetWrapper(dataset_config=dataset_configuration,
                                      model_name=model_configuration.exp_name,
                                      split_dir=split_dir, features_dir=data_dir)
    
    explainer_experiment = ExplainerExperiments(model_configuration, dataset_configuration, exp_path)
    
    explainer_experiment.run_valid(dataset, args.attribution_name, logger=logger, other={'model_path':args.model_path})
    if not os.path.exists(args.model_path):
        explainer_experiment.run_valid(dataset, args.attribution_name, logger=logger, other={'model_path':args.model_path})

    results, atom_importance, bond_importance = explainer_experiment.molecule_importance(dataset=dataset, attribution=args.attribution_name, logger=logger, other={'model_path':args.model_path}, testing=args.testing)

    logger.log('Test results: %s' % str(results))

    # print(attribution_results)

    # if args.dataset_name in ['hERG', 'CYP3A4']:
    #     attribution_results, opt_threshold = explainer_experiment.evaluate_cliffs(dataset, atom_importance, bond_importance)
    # else:
    #     binary = True if args.attribution_name == 'MCTS' else False
    #     attribution_results, opt_threshold = explainer_experiment.evaluate_attributions(dataset, atom_importance, bond_importance, binary=binary)


    # logger.log('attribution_results:' + str(attribution_results))
    # logger.log('opt_threshold:' + str(opt_threshold))

    explainer_experiment.visualization(dataset, atom_importance, bond_importance, svg_dir=svg_dir, testing=args.testing)

    # df = pd.DataFrame(
    #     {'SMILES': dataset.get_smiles_list(), 'Atom_importance': atom_importance, 'Bond_importance':bond_importance}
    # )
    # df.to_csv(os.path.join(svg_dir, "importances.csv"), index=False)
