import os

class DatasetReader:
    POSITIVE_CLASS_LABEL = 1
    NEGATIVE_CLASS_LABEL = 2
    NEUTRAL_CLASS_LABEL = 3
    DOCUMENT_CLASS_INDEX = 0
    DOCUMENT_CONTENT_INDEX = 1
    DATASETS_PATH = '../../datasets'

    def get_all_docs_with_labels_and_filenames_in_directory(self, dataset_path, dataset_include_neutral_class):
        pos_labels_and_filenames = [
            (DatasetReader.POSITIVE_CLASS_LABEL, filePath) for filePath in os.listdir(f'{dataset_path}/pos')
            ]
        neg_labels_and_filenames = [
            (DatasetReader.NEGATIVE_CLASS_LABEL, filePath) for filePath in os.listdir(f'{dataset_path}/neg')
            ]

        all_labels_and_filenames = pos_labels_and_filenames + neg_labels_and_filenames

        if dataset_include_neutral_class:
            neutral_labels_and_filenames = [
                (DatasetReader.NEUTRAL_CLASS_LABEL, filePath) for filePath in os.listdir(f'{dataset_path}/neutral')
            ]

            all_labels_and_filenames = all_labels_and_filenames + neutral_labels_and_filenames

        return all_labels_and_filenames

    def get_directory_name_for_class(self, document_class):
        if (document_class == DatasetReader.POSITIVE_CLASS_LABEL):
            return 'pos'
        elif (document_class == DatasetReader.NEGATIVE_CLASS_LABEL):
            return 'neg'
        elif (document_class == DatasetReader.NEUTRAL_CLASS_LABEL):
            return 'neutral'

    def get_document_class(self, document):
        return document[DatasetReader.DOCUMENT_CLASS_INDEX]

    def get_document_filename(self, document):
        return document[DatasetReader.DOCUMENT_CONTENT_INDEX]

    def get_document_contents(self, dataset_path, document):
        document_class = self.get_document_class(document)
        class_directory = self.get_directory_name_for_class(document_class)
        document_filename = self.get_document_filename(document)

        with open(f'{dataset_path}/{class_directory}/{document_filename}', 'r', encoding='utf-8') as document_content:
            return document_content.read()

    def get_dataset_path(self, dataset_name, dataset_type):
        dataset_path = f'{DatasetReader.DATASETS_PATH}/{dataset_name}'

        if dataset_name == 'imdb':
            # Check if IMDB dataset (or at least test and train dirs) exists in directory.
            if not os.path.isdir(f'{DatasetReader.DATASETS_PATH}/{dataset_name}/test') or \
                not os.path.isdir(f'{DatasetReader.DATASETS_PATH}/{dataset_name}/train'):
                raise FileNotFoundError('IMDB dataset is missing.')

            if dataset_type == 'train':
                dataset_path = f'{DatasetReader.DATASETS_PATH}/{dataset_name}/train'
            elif dataset_type == 'test':
                dataset_path = f'{DatasetReader.DATASETS_PATH}/{dataset_name}/test'
        else:
            # Check if dataset (or at least pos, neg, neutral dirs) exists in directory.
            if not os.path.isdir(f'{DatasetReader.DATASETS_PATH}/{dataset_name}/pos') or \
                not os.path.isdir(f'{DatasetReader.DATASETS_PATH}/{dataset_name}/neg') or \
                    not os.path.isdir(f'{DatasetReader.DATASETS_PATH}/{dataset_name}/neutral'):
                raise FileNotFoundError(f'Dataset {dataset_name} is missing.')

        return dataset_path

    def read_dataset(self, dataset_name, dataset_type, dataset_include_neutral_class = ''):
        documents = []
        dataset_path = self.get_dataset_path(dataset_name, dataset_type)
        all_documents_class_labels_and_filenames = self.get_all_docs_with_labels_and_filenames_in_directory(dataset_path, dataset_include_neutral_class)

        for documentNumber, document in enumerate(all_documents_class_labels_and_filenames):
            document_content = self.get_document_contents(dataset_path, document)
            document_class = self.get_document_class(document)
            documents.append(
                (document_class, document_content)
            )

            print(f'\r Reading {dataset_name} dataset ({dataset_type}) files {(documentNumber / len(all_documents_class_labels_and_filenames)) * 100:.2f}%...', end='')

        print('')
        return documents