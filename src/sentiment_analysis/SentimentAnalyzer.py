from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from DatasetTransformer import DatasetTransformer
from ResultsWriter import ResultsWriter

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

        # # Store most informative features (TODO).
        # if not self._args.training_set_size:
        #     self._most_informative_features = []
        #     self._most_informative_features.append(self.most_informative_feature_for_class(0))
        #     self._most_informative_features.append(self.most_informative_feature_for_class(1))
        #     if self._args.include_neutral:
        #         self._most_informative_features.append(self.most_informative_feature_for_class(2))

    def test(self):
        test_data = self._datasets.transform_test_set()
        y_true = self._datasets.true_labels
        y_pred = self._clf.predict(test_data)
        target_names = self._datasets.target_names

        # Write results to txt (including number of features and most informative features), TODO make prettier..
        result = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        # if not self._args.training_set_size:
        # self._resultsWriter.write_result(result, self._number_of_features, self._most_informative_features)
        # elif self._args.training_set_size:
        self._resultsWriter.write_result(result, self._number_of_features)

    # Inspired by this excellent answer https://stackoverflow.com/questions/30017491/problems-obtaining-most-informative-features-with-scikit-learn.
    def most_informative_feature_for_class(self, class_idx):
        feature_names = self._datasets.vectorizer.get_feature_names()
        class_label = self._datasets.target_names[class_idx]

        if self._args.algorithm == 'MNB':
            top_ten_features = sorted(zip(self._clf.feature_log_prob_[class_idx, :], feature_names))[-20:]
        elif self._args.algorithm == 'SVM':
            svm_coef = self._clf.coef_.toarray()
            top_ten_features = sorted(zip(svm_coef[class_idx], feature_names))[-20:]

        return (class_label, top_ten_features)