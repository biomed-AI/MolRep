# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Errica et al "A Fair Comparison of Graph Neural Networks for Graph classification" -> https://github.com/diningphil/gnn-comparison
"""

import os
import numpy as np

import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

import pickle
from itertools import repeat, chain
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from molrep.common.registry import registry
from molrep.processors.features import mol_to_graph_data_obj_simple


@registry.register_processor("pretraining")
class PretrainingEmbeddings(InMemoryDataset):
    def __init__(self, cfg, smiles_list, **kwargs):
        # self.transform = kwargs['transform']

        self.smiles_list = smiles_list
        self.model_name = cfg.model_cfg.name
        self.dataset_config = cfg.datasets_cfg
        self.dataset_name = self.dataset_config["name"]
        self.kwargs = kwargs

        self.features_dir = Path(registry.get_path("features_root")) / self.dataset_name
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.features_dir / (self.model_name + ".pt")

        self.smiles_col = self.dataset_config["smiles_column"]
        self.mol_id_col = self.dataset_config["id_column"]

    @property
    def dim_features(self):
        return self._dim_features

    @property
    def dim_edge_features(self):
        return self._dim_edge_features

    @property
    def max_num_nodes(self):
        return self._max_num_nodes

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.cat_dim(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_paths(self):
        return [self.features_path]

    @property
    def processed_file_names(self):
        return self.features_path

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        """
        Load and featurize data stored in a CSV file.
        """
        features_path = self.features_path

        if os.path.exists(features_path):
            dataset = torch.load(features_path)
            self._dim_features = dataset[0].x.size(1)
            self._dim_edge_features = dataset[0].edge_attr.size(1)

            # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
            self._max_num_nodes = dataset[0].max_num_nodes if hasattr(dataset[0], 'max_num_nodes') else max([da.x.shape[0] for da in dataset])

        else:
            smiles_list = self.smiles_list

            dataset = []
            if self.dataset_name == 'zinc_standard_agent':
                mol_id_list = self.kwargs['zinc_ids']

                for i in range(len(smiles_list)):
                    s = smiles_list[i]
                    try:
                        rdkit_mol = AllChem.MolFromSmiles(s)
                        if rdkit_mol != None:  # ignore invalid mol objects
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            id = int(mol_id_list[i].split('ZINC')[1].lstrip('0'))
                            data.id = torch.tensor(
                                [id])  # id here is zinc_standard_agent id value, stripped of
                            # leading zeros
                            dataset.append(data)
                    except:
                        continue

            elif self.dataset_name == 'chembl_filtered':

                folds, labels = self.kwargs["folds"], self.kwargs["labels"]
                rdkit_mol_objs = [AllChem.MolFromSmiles(s) for s in smiles_list]

                for i in range(len(rdkit_mol_objs)):
                    rdkit_mol = rdkit_mol_objs[i]
                    if rdkit_mol != None:
                        mw = Descriptors.MolWt(rdkit_mol)
                        if 50 <= mw <= 900:
                            inchi = create_standardized_mol_id(smiles_list[i])
                            if inchi != None:
                                data = mol_to_graph_data_obj_simple(rdkit_mol)
                                # manually add mol id
                                data.id = torch.tensor(
                                    [i])  # id here is the index of the mol in
                                # the dataset
                                data.y = torch.tensor(labels[i, :])
                                # fold information
                                if i in folds[0]:
                                    data.fold = torch.tensor([0])
                                elif i in folds[1]:
                                    data.fold = torch.tensor([1])
                                else:
                                    data.fold = torch.tensor([2])
                                dataset.append(data)

            else:
                raise NotImplementedError

            self._dim_features = dataset[0].x.size(1)
            self._dim_edge_features = dataset[0].edge_attr.size(1)
            self._max_num_nodes = dataset[0].max_num_nodes if hasattr(dataset[0], 'max_num_nodes') else max([da.x.shape[0] for da in dataset])
            
            # data, slices = self.collate(dataset)
            torch.save(dataset, self.processed_paths[0])
            # torch.save(dataset, features_path)
        return dataset


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def create_standardized_mol_id(smiles):
    """
    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return



def _load_chembl_with_labels_dataset(root_path):
    """
    Data from 'Large-scale comparison of machine learning methods for drug target prediction on ChEMBL'
    :param root_path: path to the folder containing the reduced chembl dataset
    :return: list of smiles, preprocessed rdkit mol obj list, list of np.array
    containing indices for each of the 3 folds, np.array containing the labels
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl

    # 1. load folds and labels
    f=open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds=pickle.load(f)
    f.close()

    f=open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat=pickle.load(f)
    sampleAnnInd=pickle.load(f)
    targetAnnInd=pickle.load(f)
    f.close()

    targetMat=targetMat
    targetMat=targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd=targetAnnInd
    targetAnnInd=targetAnnInd-targetAnnInd.min()

    folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed=targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData=targetMat.A # possible values are {-1, 0, 1}

    # 2. load structures
    f=open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr=pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')
    for i in range(len(rdkitArr)):
        print(i)
        m = rdkitArr[i]
        if m == None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in
                   preprocessed_rdkitArr]   # bc some empty mol in the
    # rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData