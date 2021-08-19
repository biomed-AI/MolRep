

# from MolRep.Utils.utils import load_checkpoint


# def load_model_and_dataset(args):
    
#     dataset_configuration = {
#                 'name': args.dataset_name,
#                 'path': args.dataset_path,
#                 'smiles_column': args.smiles_column,
#                 'target_columns': args.target_columns,
#                 'task_type': args.task_type,
#                 'metric_type': "auc" if args.task_type == 'Classification' else "rmse",
#                 'split_type': "random"
#                 }

#     dataset_wrapper = DatasetWrapper(dataset_config=dataset_configuration,
#                                      model_name=model_configuration.exp_name,
#                                      split_dir=split_dir, features_dir=data_dir,
#                                      outer_k=outer_k, inner_k=inner_k, holdout_test_size=holdout_test_size)


#     return train_loader, model

# def molecule_importance(dataloader, model, args):
    
