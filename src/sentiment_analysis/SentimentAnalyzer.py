from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from DatasetParser import DatasetParser
import sys

class SentimentAnalyzer:
    MULTINOMIAL_NAIVE_BAYES = 'MNB'
    SUPPORT_VECTOR_MACHINES = 'SVM'

    def __init__(self, args):
        self.args = args
        self.datasets = DatasetParser(self.args)

    def train(self):
        X = self.datasets.get_training_data()
        y = self.datasets.get_target_labels()
        if self.args.algorithm == SentimentAnalyzer.MULTINOMIAL_NAIVE_BAYES:
            self.clf = MultinomialNB()
        elif self.args.algorithm == SentimentAnalyzer.SUPPORT_VECTOR_MACHINES:
            self.clf = svm.SVC(kernel='linear', verbose=True, C=0.1)
 
        self.clf.fit(X, y)

    def test(self):
        test_data = self.datasets.get_test_data()
        y_true = self.datasets.get_true_labels()
        y_pred = self.clf.predict(test_data)

        target_names = self.datasets.get_target_names()
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))