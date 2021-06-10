from DatasetReader import DatasetReader

class Dataset:
    def __init__(self, dataset_name, dataset_type, dataset_include_neutral_class = ''):
        self.dataset_name = dataset_name
        self.dataset_reader = DatasetReader()
        self.dataset_labels_and_features = self.dataset_reader.read_dataset(dataset_name, dataset_type, dataset_include_neutral_class)

    # Return X (X_train or X_predict).
    def get_features(self):
        return [document[DatasetReader.DOCUMENT_CONTENT_INDEX] for document in self.dataset_labels_and_features]

    # Return y (y_target or y_true labels).
    def get_labels(self):
        return [document[DatasetReader.DOCUMENT_CLASS_INDEX] for document in self.dataset_labels_and_features]