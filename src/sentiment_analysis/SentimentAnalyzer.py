from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from dataset import DatasetTransformer
from dataset.file_handling import ResultsWriter

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