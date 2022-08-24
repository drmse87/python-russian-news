import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import Lemmatizer

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
        if not os.path.isdir(f'{self._dataset_path}/{class_pathname}'):
            raise FileNotFoundError(f'Error locating {class_pathname} in {self._dataset_path}.')
        filenames = [(class_label, filePath) for filePath in os.listdir(f'{self._dataset_path}/{class_pathname}')]

        # Sometimes training size should be reduced...
        if self._args.training_set_size and self._dataset_path == self._args.training_set:
            number_of_classes = 2 if not self._args.include_neutral else 3
        
            filenames = filenames[:self._args.training_set_size // number_of_classes]

        return filenames

    def get_all_filenames(self):
        pos_labels_and_filenames = self.get_filenames_in_classpath(DatasetReader.POSITIVE_CLASS_LABEL, DatasetReader.POSITIVE_CLASS_PATHNAME)
        neg_labels_and_filenames = self.get_filenames_in_classpath(DatasetReader.NEGATIVE_CLASS_LABEL, DatasetReader.NEGATIVE_CLASS_PATHNAME)
        all_labels_and_filenames = pos_labels_and_filenames + neg_labels_and_filenames

        if self._args.include_neutral:
            neutral_labels_and_filenames = self.get_filenames_in_classpath(DatasetReader.NEUTRAL_CLASS_LABEL, DatasetReader.NEUTRAL_CLASS_PATHNAME)
            all_labels_and_filenames = all_labels_and_filenames + neutral_labels_and_filenames

        return all_labels_and_filenames     

class DatasetTransformer:
    def __init__(self, args):
        self._training_set = Dataset(args.training_set, args)
        self._test_set = Dataset(args.test_set, args)

        # Set target names.
        self._target_names = ['Positive', 'Negative']
        if args.include_neutral:
            self._target_names.append('Neutral')

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
        return self._training_set.labels

    @property
    def true_labels(self):
        return self._test_set.labels

    @property
    def vectorizer(self):
        return self._vectorizer

    def transform_training_set(self):
        return self._vectorizer.fit_transform(self._training_set.documents)
        
    def transform_test_set(self):
        return self._vectorizer.transform(self._test_set.documents)