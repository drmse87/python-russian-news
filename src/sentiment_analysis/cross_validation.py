from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import Lemmatizer
from data import Dataset
import numpy as np

# Crossvalidate both training sets.
for current_training_set in [
    Dataset('imdb', 'train'), 
    Dataset('good_bad_neutral_news', 'train')
    ]:
    for current_vectorizer in [ # ngram_range=(1, 1) for unigram or (2, 2) for bigram.
        CountVectorizer(tokenizer=Lemmatizer(current_training_set)), 
        TfidfVectorizer(tokenizer=Lemmatizer(current_training_set))
        ]:
        current_stage = f'{current_training_set.dataset_name}, {current_vectorizer.__class__.__name__}'

        # Use 10-fold cross validation (MNB).
        X = current_vectorizer.fit_transform(current_training_set.get_features())
        y = current_training_set.get_labels()
        accuracy = cross_val_score(MultinomialNB(), X, y, cv=KFold(n_splits=10, shuffle=True))
        avg_accuracy = np.mean(accuracy)
        print(f'Accuracy for each fold ({current_stage}): {accuracy}')
        print(f'Average accuracy ({current_stage}): {avg_accuracy}')

        # Use grid search to find best C parameter (SVM) with linear kernel.
        clf = GridSearchCV(SVC(), verbose=3, param_grid={'C': [0.1, 1, 10], 'kernel': ['linear']}, cv=KFold(n_splits=10, shuffle=True))
        clf.fit(X, y)
        print(f'Best C ({current_stage}): {clf.best_estimator_.get_params()["C"]}')