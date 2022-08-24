from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from data import DatasetTransformer
from util import ResultsWriter
import argparse

class SentimentAnalyzer:
    def __init__(self, args):
        self._args = args
        self._datasets = DatasetTransformer(args)
        self._resultsWriter = ResultsWriter(args)

    def train(self):
        X = self._datasets.transform_training_set()
        y = self._datasets.target_labels
        if self._args.algorithm == 'MNB':
            self._clf = MultinomialNB()
        elif self._args.algorithm == 'SVM':
            self._clf = svm.SVC(kernel='linear', verbose=True, C=0.1)
        self._clf.fit(X, y)

        # Store number of features.
        if self._args.algorithm == 'MNB':
            self._number_of_features = self._clf.n_features_
        elif self._args.algorithm == 'SVM':
            self._number_of_features = self._clf.coef_.shape[-1]

    def test(self):
        test_data = self._datasets.transform_test_set()
        y_true = self._datasets.true_labels
        y_pred = self._clf.predict(test_data)
        target_names = self._datasets.target_names

        # Write results to txt (including number of features)
        result = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        self._resultsWriter.write_result(result, self._number_of_features)

def main():
    parser = argparse.ArgumentParser(description='Train a MNB or SVM Sentiment Analyzer.')
    parser.add_argument('training_set', metavar='Training set directory', help='Training set directory.')
    parser.add_argument('test_set', metavar='Test set directory', help='Test set directory.')
    parser.add_argument('-a', '--algorithm', dest='algorithm', default='MNB', choices=['MNB', 'SVM'], help='Algorithm.')
    parser.add_argument('-v', '--vectorizer', dest='vectorizer', default='tf-idf', choices=['tf-idf', 'count'], help='Vectorizer (feature [count], or fractional [tf-idf] count).')
    parser.add_argument('-s', '--size', '--training_set_size', dest='training_set_size', type=int, help='Training set size.')
    parser.add_argument('-ng', '--ngram', '--ngram_length', dest='ngram_length', default='unigram', help='N-gram length.', choices=['unigram', 'bigram', 'trigram'])
    parser.add_argument('-ne', '--neutral', dest='include_neutral', default=False, help='Include neutral class?', action=argparse.BooleanOptionalAction)
    parser.add_argument('-sw', '--stopwords', dest='use_stopwords', default=True, help='Use stop words?', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    print(f'Starting sentiment analysis with: {args}')
    s = SentimentAnalyzer(args)
    print('Starting training.')
    s.train()
    print('Starting testing.')
    s.test()

if __name__ == '__main__':
    main()

