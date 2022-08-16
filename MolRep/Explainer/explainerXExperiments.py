

import os
import random
import sklearn
import collections

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs

from MolRep.Models.losses import get_loss_func
from MolRep.Explainer.explainerXNetWrapper import ExplainerXNetWrapper
from MolRep.Models.schedulers import build_lr_scheduler

from MolRep.Utils.config_from_dict import Config
from MolRep.Explainer.Metrics import attribution_metric as att_metrics
from MolRep.Utils.utils import *


GREEN_COL = (0, 1, 0)
RED_COL = (1, 0, 0)


class XAIKeys:

    def __init__(self, fragment):
        self.fragment = fragment

    def GenXAIKeys(self, mol):
        maccsXAIKeys = [(None, 0)] * len(self.fragment)

        for idx, smarts in enumerate(self.fragment):
            patt, count = smarts, 0
            if patt != '?':
                sma = Chem.MolFromSmarts(patt)
                if not sma:
                    print('SMARTS parser error for key %s' % (patt))
                else:
                    maccsXAIKeys[idx] = sma, count

        res = DataStructs.SparseBitVect(len(maccsXAIKeys)+1)
        for i, (patt, count) in enumerate(maccsXAIKeys):
            if patt is not None:
                if count == 0:
                    res[i + 1] = mol.HasSubstructMatch(patt)
                else:
                    matches = mol.GetSubstructMatches(patt)
                    if len(matches) > count:
                        res[i + 1] = 1

        return res


class ExplainerXExperiments:

    def __init__(self, model_configuration, dataset_config, exp_path, subgraph_path=None):
        self.model_config = Config.from_dict(model_configuration) if isinstance(model_configuration, dict) else model_configuration
        self.dataset_config = dataset_config
        self.exp_path = exp_path
        self.subgraph_path = subgraph_path
        self.fragments = None
        if self.subgraph_path is not None:
            self.fragments = self.get_fragments(subgraph_path=self.subgraph_path, topk=self.model_config['topk'])


        if not os.path.exists(exp_path):
            os.makedirs(exp_path)


    def get_fragments(self, subgraph_path=None, eps=0.15, topk=20):
        subgraph_path = self.subgraph_path if subgraph_path is None else subgraph_path
        assert subgraph_path is not None

        d = np.load(self.subgraph_path, allow_pickle = True).item()

        selected_mol = []
        for idx, m in enumerate(d.keys()):
            selected_atom_idx = []
            for i in list(np.where(d[m] > eps)[0]):
                selected_atom_idx.append(int(i))
            for i in list(np.where(d[m] < -eps)[0]):
                selected_atom_idx.append(int(i))
                
            mol = Chem.MolFromSmiles(m)
            if len(selected_atom_idx)>0:
                selected_smi = Chem.MolFragmentToSmarts(mol, selected_atom_idx)
                selected_mol.append(selected_smi)
        
        selected_fragment = []
        for smi_list in selected_mol:
            smis = smi_list.split('.')
            for smi in smis:
                if len(smi) <= 2:
                    continue
                else:
                    selected_fragment.append(smi)

        if topk is None:
            selected_fragment = list(set(selected_fragment))
            print('Number of subgraph rule: ', len(selected_fragment))
            return selected_fragment

        else:
            num_dict = {}
            for i in range(len(selected_fragment)):
                if selected_fragment[i] in num_dict:
                    num_dict[selected_fragment[i]] = num_dict[selected_fragment[i]] + 1
                else:
                    num_dict[selected_fragment[i]] = 1
            sorted_fragment = sorted(num_dict.items(),key=lambda e:e[1],reverse=True)
            selected_fragment = [i for i,n in sorted_fragment[:topk]]
            print('Number of subgraph rule: ', len(selected_fragment))
            return selected_fragment



    def run_valid(self, dataset, logger, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR TRAINING OR EARLY STOPPING!
        :return: (training accuracy, test accuracy)
        """
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True
        subgraph_list = self.fragments

        model_class = self.model_config.model
        optim_class = self.model_config.optimizer
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        loss_fn = get_loss_func(self.dataset_config['task_type'], self.model_config.exp_name)
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True


        train_loader, scaler = dataset.get_train_loader(self.model_config['batch_size'],
                                                        shuffle=shuffle)

        model = model_class(dim_features=dataset.dim_features, dim_target=dataset.dim_target, model_configs=self.model_config, dataset_configs=self.dataset_config, subgraph_list=subgraph_list)
        net = ExplainerXNetWrapper(model, dataset_configs=self.dataset_config, model_config=self.model_config,
                                  loss_function=loss_fn)
        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])
        scheduler = build_lr_scheduler(optimizer, model_configs=self.model_config, num_samples=dataset.num_samples)

        train_loss, train_metric, _, _, _, _, _ = net.train(train_loader=train_loader,
                                                            optimizer=optimizer, scheduler=scheduler,
                                                            clipping=clipping, scaler=scaler,
                                                            early_stopping=stopper_class,
                                                            logger=logger)

        if other is not None and 'model_path' in other.keys():
            save_checkpoint(path=other['model_path'], model=model, scaler=scaler)

        return train_metric

    def run_test(self, dataset, logger, testing=True, other=None):

        subgraph_list = self.fragments
        model_class = self.model_config.model
        loss_fn = get_loss_func(self.dataset_config['task_type'], self.model_config.exp_name)
        model = model_class(dim_features=dataset.dim_features, dim_target=dataset.dim_target, model_configs=self.model_config, dataset_configs=self.dataset_config, subgraph_list=subgraph_list)

        assert 'model_path' in other.keys()
        model = load_checkpoint(path=other['model_path'], model=model)
        scaler, features_scaler = load_scalers(path=other['model_path'])
        net = ExplainerXNetWrapper(model, dataset_configs=self.dataset_config, model_config=self.model_config,
                                   loss_function=loss_fn)

        if testing:
            test_loader = dataset.get_test_loader(batch_size=50)
        else:
            test_loader = dataset.get_all_dataloader()
        y_preds, y_labels, results = net.explainer(test_loader=test_loader, scaler=scaler, logger=logger)
        
        oof = pd.DataFrame({'SMILES': dataset.get_smiles_list(testing)})
        oof['preds'] = [p[0].index(max(p[0])) for p in y_preds]
        oof['labels'] = [y[0] for y in y_labels]
        oof.to_csv(Path(other['model_path']).parent / f'{self.model_config.exp_name}_explained+_oof.csv', index=False)
        
        # atom_imp_dict = dict(zip(dataset.get_smiles_list(testing), atom_importance))
        # np.save(Path(other['model_path']).parent / f'{self.model_config.exp_name}_explained+_atom_importance.npy', atom_imp_dict)
        return results
