import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import Lemmatizer

class Dataset:
    def __init__(self, dataset_path, args):
        self._dataset_path = dataset_path

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f'Error locating {dataset_path}.')
        
        dataset_dir_contents = os.listdir(dataset_path)
        if not dataset_dir_contents:
            raise FileNotFoundError(f'No labels (folders) found in {dataset_path}.')
        else:
            self._labels = dataset_dir_contents

        self._documents = self.read_dataset()

    @property
    def dataset_path(self):
        return self._dataset_path

    @property
    def documents(self):
        return [document.contents for document in self._documents]

    @property
    def document_labels(self):
        return [document.label for document in self._documents]

    @property
    def labels(self):
        return self._labels

    def read_dataset(self):
        fileNames_with_labels = [(label, filePath) for label in self._labels for filePath in os.listdir(f'{self._dataset_path}/{label}')]
        documents = []
  
        for fileNumber, file in enumerate(fileNames_with_labels):
            file_label = file[0]
            filename = file[1]

            file_contents = ''
            with open(f'{self._dataset_path}/{file_label}/{filename}', 'r', encoding='utf-8') as input_file:
                file_contents = input_file.read()

            documents.append(Document(file_label, file_contents))

            print(f'\r Reading {fileNumber}/{len(fileNames_with_labels)} files in {self._dataset_path} {(fileNumber / len(fileNames_with_labels)) * 100:.2f}%...', end='')

        print('')
        return documents

class Document:
    def __init__(self, label, contents):
        self._label = label
        self._contents = contents

    @property
    def label(self):
        return self._label

    @property
    def contents(self):
        return self._contents

class DatasetTransformer:
    def __init__(self, args):
        self._training_set = Dataset(args.training_set, args)
        self._test_set = Dataset(args.test_set, args)

        if sorted(self._training_set.labels) != sorted(self._test_set.labels):
            raise ValueError("Labels/classes mismatch in training and test set.")

        # Set target names.
        self._target_names = self._training_set.labels

        # Set n-gram length.
        ngram_length = ()
        if args.ngram_length == 'unigram':
            ngram_length = (1, 1)
        elif args.ngram_length == 'bigram':
            ngram_length = (2, 2)
        elif args.ngram_length == 'trigram':
            ngram_length = (3, 3)

        # Set vectorizer.
        if args.vectorizer == 'tf-idf':
            self._vectorizer = TfidfVectorizer(tokenizer=Lemmatizer(args), ngram_range=ngram_length)
        elif args.vectorizer == 'count':
            self._vectorizer = CountVectorizer(tokenizer=Lemmatizer(args), ngram_range=ngram_length)

    @property
    def target_names(self):
        return self._target_names

    @property
    def target_labels(self):
        return self._training_set.document_labels

    @property
    def true_labels(self):
        return self._test_set.document_labels

    @property
    def vectorizer(self):
        return self._vectorizer

    def transform_training_set(self):
        return self._vectorizer.fit_transform(self._training_set.documents)
        
    def transform_test_set(self):
        return self._vectorizer.transform(self._test_set.documents)