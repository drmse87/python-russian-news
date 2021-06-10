from Dataset import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from Lemmatizer import Lemmatizer

class DatasetParser:
    def __init__(self, args):
        self.training_set = Dataset(args.training_set, args)
        self.test_set = Dataset(args.test_set, args)

        # Prepare target names.
        self.target_names = ['Positive', 'Negative']
        if args.include_neutral:
            self.target_names.append('Neutral')

        # Prepare n-gram length.
        ngram_length = ()
        if args.ngram_length == 'unigram':
            ngram_length = (1, 1)
        elif args.ngram_length == 'bigram':
            ngram_length = (2, 2)
        elif args.ngram_length == 'trigram':
            ngram_length = (3, 3)

        # Prepare vectorizer (also set n-gram range).
        if args.vectorizer == 'tf-idf':
            self.vectorizer = TfidfVectorizer(tokenizer=Lemmatizer(args), ngram_range=ngram_length)
        elif args.vectorizer == 'count':
            self.vectorizer = CountVectorizer(options)

    def get_training_data(self):
        return self.vectorizer.fit_transform(self.training_set.get_docs())
        
    def get_test_data(self):
        return self.vectorizer.transform(self.test_set.get_docs())

    def get_target_labels(self):
        return self.training_set.get_labels()

    def get_true_labels(self):
        return self.test_set.get_labels()

    def get_target_names(self):
        return self.target_names
