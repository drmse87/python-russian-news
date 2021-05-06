from Dataset import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from Lemmatizer import Lemmatizer

class TrainTestDatasetParser:
    VALID_TRAINING_SETS = ['imdb', 'good_neutral_bad_news']
    VALID_TEST_SETS = ['imdb', 'russian_news']
    TARGET_NAMES_TWO_CLASSES = ['Positive', 'Negative']
    TARGET_NAMES_THREE_CLASSES = ['Positive', 'Negative', 'Neutral']

    def __init__(self, training_set, test_set):
        # Check dataset arguments are valid.
        if training_set not in TrainTestDatasetParser.VALID_TRAINING_SETS or \
            test_set not in TrainTestDatasetParser.VALID_TEST_SETS:
            raise ValueError('Invalid dataset.')

        # Check dataset combinations and initialize datasets.
        if (training_set == 'imdb' and test_set == 'imdb'):
            self.training_set = Dataset('imdb', 'train')
            self.test_set = Dataset('imdb', 'test')
            self.target_names = TrainTestDatasetParser.TARGET_NAMES_TWO_CLASSES
        elif (training_set == 'imdb' and test_set == 'russian_news'):
            self.training_set = Dataset('imdb', 'train')
            self.test_set = Dataset('russian_news', 'test')
            self.target_names = TrainTestDatasetParser.TARGET_NAMES_TWO_CLASSES
        elif (training_set == 'good_neutral_bad_news' and test_set == 'russian_news'):
            # Include neutral class for these dataset combinations.
            self.training_set = Dataset('good_neutral_bad_news', 'train', True)
            self.test_set = Dataset('russian_news', 'test', True)
            self.target_names = TrainTestDatasetParser.TARGET_NAMES_THREE_CLASSES
        else:
            raise ValueError("Invalid dataset combination.")

        # Pass training set name to tokenizer, to add dataset specific stop words.
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=Lemmatizer(training_set))

    def get_training_data(self):
        return self.tfidf_vectorizer.fit_transform(
                self.training_set.get_features()
            )
        
    def get_test_data(self):
        return self.tfidf_vectorizer.transform(
                self.test_set.get_features()
            )

    def get_target_labels(self):
        return self.training_set.get_labels()

    def get_true_labels(self):
        return self.test_set.get_labels()

    def get_target_names(self):
        return self.target_names
