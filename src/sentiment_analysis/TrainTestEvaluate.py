from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from TrainTestDatasetParser import TrainTestDatasetParser
import sys

MULTINOMIAL_NAIVE_BAYES = 'mnb'
SUPPORT_VECTOR_MACHINES = 'svm'

# Read arguments.
if len(sys.argv) != 4:
    raise ValueError('Invalid number of arguments (training_data test_data algorithm).')
training_data = sys.argv[1]
test_data = sys.argv[2]
algorithm = sys.argv[3]

# Initialize datasets.
train_test_dataset_parser = TrainTestDatasetParser(training_data, test_data)

# Train.
X = train_test_dataset_parser.get_training_data()
y = train_test_dataset_parser.get_target_labels()
if algorithm == MULTINOMIAL_NAIVE_BAYES:
    clf = MultinomialNB()
elif algorithm == SUPPORT_VECTOR_MACHINES:
    clf = svm.SVC(kernel='linear', verbose=True, C=0.1)
else:
    raise ValueError('Invalid algorithm.')    
clf.fit(X, y)

# Predict.
test_data = train_test_dataset_parser.get_test_data()
y_true = train_test_dataset_parser.get_true_labels()
y_pred = clf.predict(test_data)

# Evaluate/score.
target_names = train_test_dataset_parser.get_target_names()
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))