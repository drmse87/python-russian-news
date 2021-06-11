from DatasetReader import DatasetReader

class Dataset:
    def __init__(self, dataset_path, args):
        self._dataset_path = dataset_path
        self._documents = DatasetReader(dataset_path, args).read_dataset()

    @property
    def dataset_path(self):
        return self._dataset_path

    @property
    def documents(self):
        return [document.contents for document in self._documents]

    @property
    def labels(self):
        return [document.label for document in self._documents]
