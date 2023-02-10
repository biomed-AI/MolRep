


class DataLoaderWrapper:

    def __init__(self, dataset, outer_k=None, inner_k=None):
        self.outer_k = outer_k
        self.inner_k = inner_k
        self.dataset = dataset

    def set_inner_k(self, k):
        self.inner_k = k
    
    def set_outer_k(self, k):
        self.outer_k = k

    @property
    def get_dataset(self):
        return self.dataset

    @property
    def task_type(self):
        return self.dataset.task_type

    @property
    def num_samples(self):
        return self.dataset.num_samples

    def get_train_val(self, dataset, batch_size, shuffle=True):
        return dataset.get_model_selection_fold(self.outer_k, self.inner_k, batch_size, shuffle)

    def get_test(self, dataset, batch_size, shuffle=False):
        return dataset.get_test_fold(self.outer_k, batch_size, shuffle)
