from Dataset import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from Lemmatizer import Lemmatizer

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
            self.vectorizer = TfidfVectorizer(tokenizer=Lemmatizer(args), ngram_range=ngram_length)
        elif args.vectorizer == 'count':
            self.vectorizer = CountVectorizer(tokenizer=Lemmatizer(args), ngram_range=ngram_length)

    @property
    def target_names(self):
        return self._target_names

    @property
    def target_labels(self):
        return self._training_set.labels

    @property
    def true_labels(self):
        return self._test_set.labels

    def transform_training_set(self):
        return self.vectorizer.fit_transform(self._training_set.documents)
        
    def transform_test_set(self):
        return self.vectorizer.transform(self._test_set.documents)


