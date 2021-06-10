import os

class Dataset:
    POSITIVE_CLASS_LABEL = 1
    NEGATIVE_CLASS_LABEL = 2
    NEUTRAL_CLASS_LABEL = 3
    DOCUMENT_CLASS_INDEX = 0
    DOCUMENT_CONTENT_INDEX = 1

    def __init__(self, dataset_path, args):
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f'Error finding {dataset_path}.')

        self.dataset_path = dataset_path
        self.args = args
        self.dataset_contents = self.read_dataset_contents()

    def get_docs(self):
        return [document[Dataset.DOCUMENT_CONTENT_INDEX] for document in self.dataset_contents]

    def get_labels(self):
        return [document[Dataset.DOCUMENT_CLASS_INDEX] for document in self.dataset_contents]

    def read_dataset_contents(self):
        documents = []
        dir_contents = self.read_dir_contents()

        for documentNumber, document in enumerate(dir_contents):
            document_content = self.read_doc_contents(document)
            document_class = document[Dataset.DOCUMENT_CLASS_INDEX]
            documents.append((document_class, document_content))

            print(f'\r Reading files in {self.dataset_path} {(documentNumber / len(dir_contents)) * 100:.2f}%...', end='')

        print('')
        return documents

    def read_dir_contents(self):
        pos_labels_and_filenames = [(Dataset.POSITIVE_CLASS_LABEL, filePath) for filePath in os.listdir(f'{self.dataset_path}/pos')]
        neg_labels_and_filenames = [(Dataset.NEGATIVE_CLASS_LABEL, filePath) for filePath in os.listdir(f'{self.dataset_path}/neg')]
        all_labels_and_filenames = pos_labels_and_filenames + neg_labels_and_filenames

        if self.args.include_neutral:
            neutral_labels_and_filenames = [(Dataset.NEUTRAL_CLASS_LABEL, filePath) for filePath in os.listdir(f'{self.dataset_path}/neutral')]
            all_labels_and_filenames = all_labels_and_filenames + neutral_labels_and_filenames

        # Sometimes training size should be reduced.
        if self.dataset_path == self.args.training_set:
            all_labels_and_filenames =  all_labels_and_filenames[:self.args.size]
            print(len(all_labels_and_filenames))

        return all_labels_and_filenames

    def read_doc_contents(self, document):
        document_class = document[Dataset.DOCUMENT_CLASS_INDEX]
        class_directory = ''
        if (document_class == Dataset.POSITIVE_CLASS_LABEL):
            class_directory = 'pos'
        elif (document_class == Dataset.NEGATIVE_CLASS_LABEL):
            class_directory = 'neg'
        elif (document_class == Dataset.NEUTRAL_CLASS_LABEL):
            class_directory = 'neutral'
        document_filename = document[Dataset.DOCUMENT_CONTENT_INDEX]

        with open(f'{self.dataset_path}/{class_directory}/{document_filename}', 'r', encoding='utf-8') as document_content:
            return document_content.read()
