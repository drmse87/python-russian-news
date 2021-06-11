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

        filenames_with_labels = self.read_filenames_and_labels()

        for fileNumber, file in enumerate(filenames_with_labels):
            # Get file path depending on class.
            document_label = file[DatasetReader.DOCUMENT_CLASS_INDEX]
            label_pathname = ''
            if (document_label == DatasetReader.POSITIVE_CLASS_LABEL):
                label_pathname = DatasetReader.POSITIVE_CLASS_PATHNAME
            elif (document_label == DatasetReader.NEGATIVE_CLASS_LABEL):
                label_pathname = DatasetReader.NEGATIVE_CLASS_PATHNAME
            elif (document_label == DatasetReader.NEUTRAL_CLASS_LABEL):
                label_pathname = DatasetReader.NEUTRAL_CLASS_PATHNAME

            # Read each file's contents.
            document_filename = file[DatasetReader.DOCUMENT_CONTENT_INDEX]
            document_contents = ''
            with open(f'{self._dataset_path}/{label_pathname}/{document_filename}', 'r', encoding='utf-8') as document_contents_from_file:
                document_contents = document_contents_from_file.read()

            documents.append(Document(document_label, document_contents))

            print(f'\r Reading {fileNumber}/{len(filenames_with_labels)} files in {self._dataset_path} {(fileNumber / len(filenames_with_labels)) * 100:.2f}%...', end='')

        print('')
        return documents

    def read_filenames_and_labels(self):
        # Sometimes training size should be reduced.
        if self._dataset_path == self._args.training_set:
            print('hej')
        if self._args.training_set_size:
            print('yeah')      

        pos_labels_and_filenames = [
            (DatasetReader.POSITIVE_CLASS_LABEL, filePath) 
                for filePath in os.listdir(f'{self._dataset_path}/{DatasetReader.POSITIVE_CLASS_PATHNAME}')
            ]
        neg_labels_and_filenames = [
            (DatasetReader.NEGATIVE_CLASS_LABEL, filePath) 
                for filePath in os.listdir(f'{self._dataset_path}/{DatasetReader.NEGATIVE_CLASS_PATHNAME}')
            ]
        all_labels_and_filenames = pos_labels_and_filenames + neg_labels_and_filenames

        if self._args.include_neutral:
            neutral_labels_and_filenames = [
                (DatasetReader.NEUTRAL_CLASS_LABEL, filePath) 
                    for filePath in os.listdir(f'{self._dataset_path}/{DatasetReader.NEUTRAL_CLASS_PATHNAME}')
                ]
            all_labels_and_filenames = all_labels_and_filenames + neutral_labels_and_filenames

        return all_labels_and_filenames     