











from molrep.models.scalers import StandardScaler
from molrep.common.utils import worker_init



class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 2,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        dataset: The :class:`MoleculeDataset` containing the molecules to load.
        batch_size: Batch size.
        num_workers: Number of workers used to build batches.
        class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        shuffle: Whether to shuffle the data.
        seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )
        
        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout,
            worker_init_fn=worker_init,
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def smiles(self):
        return [self._dataset[index].smiles for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()



def _construct_dataset(smiles_all, x_all, y_all):
    """Construct a MolDataset object from the provided data.
    Args:
        x_all (list): A list of molecule features.
        y_all (list): A list of the corresponding labels.
    Returns:
        A MolDataset object filled with the provided data.
    """
    dataset = MoleculeDataset([
                MoleculeDatapoint(
                    smiles=smiles,
                    targets=targets,
                    features=x_all[i][0] if x_all[i][0] is not None else None,
                    atom_features=x_all[i][1] if x_all[i][1] is not None else None,
                    atom_descriptors=x_all[i][2] if x_all[i][2] is not None else None,
                ) for i, (smiles, targets) in enumerate(zip(smiles_all, y_all))
    ])

    return dataset

def _construct_dataloader(data_set, batch_size, shuffle=True, num_workers=0, seed=0, class_balance=False):
    """Construct a data loader for the provided data.
    Args:
        data_set (): 
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.
    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    if data_set is not None:
        loader = MoleculeDataLoader(
                    dataset=data_set,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    class_balance=class_balance,
                    shuffle=shuffle,
                    seed=seed,
                )
    else:
        loader = None
    return loader

def MPNN_construct_dataset(features_path, train_idxs=None, valid_idxs=None, test_idxs=None):
    # smiles_all, x_all, y_all = pickle.load(open(features_path, "rb"))
    dataset = torch.load(features_path)
    smiles_all, x_all, y_all = dataset["smiles_all"], dataset["x_all"], dataset["y_all"]

    trainset = _construct_dataset(np.array(smiles_all)[train_idxs], np.array(x_all)[train_idxs], np.array(y_all)[train_idxs]) if train_idxs is not None else None
    validset = _construct_dataset(np.array(smiles_all)[valid_idxs], np.array(x_all)[valid_idxs], np.array(y_all)[valid_idxs]) if valid_idxs is not None else None
    testset = _construct_dataset(np.array(smiles_all)[test_idxs], np.array(x_all)[test_idxs], np.array(y_all)[test_idxs]) if test_idxs is not None else None
    return trainset, validset, testset

def MPNN_construct_dataloader(trainset=None, validset=None, testset=None, batch_size=1, shuffle=True, task_type='classification', seed=0, features_scaling=True):

    if features_scaling and trainset is not None:
        features_scaler = trainset.normalize_features(replace_nan_token=0)
        if validset is not None:
            validset.normalize_features(features_scaler)
        if testset is not None:
            testset.normalize_features(features_scaler)
    else:
        features_scaler = None

    if task_type == 'regression' and trainset is not None:
        train_targets = trainset.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        trainset.set_targets(scaled_targets)
    else:
        scaler = None

    return _construct_dataloader(trainset, batch_size, shuffle=shuffle, seed=seed), \
           _construct_dataloader(validset, batch_size, shuffle=False, seed=seed), \
           _construct_dataloader(testset, batch_size, shuffle=False, seed=seed), \
           features_scaler, scaler






class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 2,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 follow_batch: list = ['x'],
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0

        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            # collate_fn=construct_molecule_batch,
            collate_fn=lambda data_list: Batch.from_data_list(
                                                    data_list, follow_batch),
            multiprocessing_context=self._context,
            timeout=self._timeout,
            worker_init_fn=worker_init,
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].y for index in self._sampler]

    @property
    def smiles(self):
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].smiles for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()



def _construct_dataset(dataset, indices):
    return MoleculeDataset([dataset[idx] for idx in indices])

def _construct_dataloader(dataset, batch_size, shuffle, seed=0, num_workers=0, class_balance=False):
    """Construct a data loader for the provided data.
    Args:
        data_set (): 
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.
    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    if dataset is not None:
        loader = MoleculeDataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    class_balance=class_balance,
                    shuffle=shuffle,
                    seed=seed,
                )
    else: 
        loader = None
    return loader

def Graph_construct_dataset(features_path, train_idxs=None, valid_idxs=None, test_idxs=None):
    dataset = torch.load(features_path)

    trainset = _construct_dataset(dataset, train_idxs) if train_idxs is not None else None
    validset = _construct_dataset(dataset, valid_idxs) if valid_idxs is not None else None
    testset = _construct_dataset(dataset, test_idxs) if test_idxs is not None else None
    return trainset, validset, testset

def Graph_construct_dataloader(trainset=None, validset=None, testset=None, batch_size=1, shuffle=True, seed=0, task_type='classification', features_scaling=True):

    if features_scaling and trainset is not None:
        features_scaler = trainset.normalize_features(replace_nan_token=0)
        if validset is not None:
            validset.normalize_features(features_scaler)
        if testset is not None:
            testset.normalize_features(features_scaler)
    else:
        features_scaler = None

    if task_type == 'regression' and trainset is not None:
        train_targets = trainset.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        trainset.set_targets(scaled_targets)
    else:
        scaler = None

    return _construct_dataloader(trainset, batch_size, shuffle, seed), \
           _construct_dataloader(validset, batch_size, False, seed), \
           _construct_dataloader(testset, batch_size, False, seed), \
           features_scaler, scaler