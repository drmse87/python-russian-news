from Dataset import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from Lemmatizer import Lemmatizer

class DatasetParser:
    def __init__(self, args):
        self.training_set = Dataset(args.training_set, args)
        self.test_set = Dataset(args.test_set, args)

        self.target_names = ['Positive', 'Negative']
        if args.include_neutral:
            self.target_names.append('Neutral')

        # TODO! Pass training set name to tokenizer, to add dataset specific stop words.
        if args.vectorizer == 'tf-idf':
            self.vectorizer = TfidfVectorizer(tokenizer=Lemmatizer())
        elif args.vectorizer == 'count':
            self.vectorizer = CountVectorizer(tokenizer=Lemmatizer())

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
