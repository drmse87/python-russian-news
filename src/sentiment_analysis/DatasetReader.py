import os
from Document import Document

class DatasetReader:
    POSITIVE_CLASS_LABEL = 1
    POSITIVE_CLASS_PATHNAME = 'pos'
    NEGATIVE_CLASS_LABEL = 2
    NEGATIVE_CLASS_PATHNAME = 'neg'
    NEUTRAL_CLASS_LABEL = 3
    NEUTRAL_CLASS_PATHNAME = 'neutral'
    DOCUMENT_CLASS_INDEX = 0
    DOCUMENT_CONTENT_INDEX = 1

    def __init__(self, dataset_path, args):
        self._dataset_path = dataset_path
        self._args = args
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f'Error locating {dataset_path}.')

    def read_dataset(self):
        documents = []

        filenames_with_labels = self.get_all_filenames()
        for fileNumber, file in enumerate(filenames_with_labels):
            file_label = file[DatasetReader.DOCUMENT_CLASS_INDEX]
            label_pathname = self.get_pathname_for_class(file_label)
            filename = file[DatasetReader.DOCUMENT_CONTENT_INDEX]

            file_contents = ''
            with open(f'{self._dataset_path}/{label_pathname}/{filename}', 'r', encoding='utf-8') as input_file:
                file_contents = input_file.read()

            documents.append(Document(file_label, file_contents))

            print(f'\r Reading {fileNumber}/{len(filenames_with_labels)} files in {self._dataset_path} {(fileNumber / len(filenames_with_labels)) * 100:.2f}%...', end='')

        print('')
        return documents

    def get_pathname_for_class(self, file_label):
        if (file_label == DatasetReader.POSITIVE_CLASS_LABEL):
            return DatasetReader.POSITIVE_CLASS_PATHNAME
        elif (file_label == DatasetReader.NEGATIVE_CLASS_LABEL):
            return DatasetReader.NEGATIVE_CLASS_PATHNAME
        elif (file_label == DatasetReader.NEUTRAL_CLASS_LABEL):
            return DatasetReader.NEUTRAL_CLASS_PATHNAME

    def get_filenames_in_classpath(self, class_label, class_pathname):
        filenames = [(class_label, filePath) for filePath in os.listdir(f'{self._dataset_path}/{class_pathname}')]

        # Sometimes training size should be reduced...
        if self._args.training_set_size and self._dataset_path == self._args.training_set:
            number_of_classes = 2 if not self._args.include_neutral else 3
        
            filenames = filenames[:self._args.training_set_size // number_of_classes]
            print(filenames)

        return filenames

    def get_all_filenames(self):
        pos_labels_and_filenames = self.get_filenames_in_classpath(DatasetReader.POSITIVE_CLASS_LABEL, DatasetReader.POSITIVE_CLASS_PATHNAME)
        neg_labels_and_filenames = self.get_filenames_in_classpath(DatasetReader.NEGATIVE_CLASS_LABEL, DatasetReader.NEGATIVE_CLASS_PATHNAME)
        all_labels_and_filenames = pos_labels_and_filenames + neg_labels_and_filenames

        if self._args.include_neutral:
            neutral_labels_and_filenames = self.get_filenames_in_classpath(DatasetReader.NEUTRAL_CLASS_LABEL, DatasetReader.NEUTRAL_CLASS_PATHNAME)
            all_labels_and_filenames = all_labels_and_filenames + neutral_labels_and_filenames

        return all_labels_and_filenames     