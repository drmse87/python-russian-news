from SentimentAnalyzer import SentimentAnalyzer
import argparse

parser = argparse.ArgumentParser(description='Train a MNB or SVM Sentiment Analyzer.')
parser.add_argument('training_set', metavar='Training set directory', help='Training set directory.')
parser.add_argument('test_set', metavar='Test set directory', help='Test set directory.')
parser.add_argument('-a', '--algorithm', dest='algorithm', default='MNB', choices=['MNB', 'SVM'], help='Algorithm.')
parser.add_argument('-v', '--vectorizer', dest='vectorizer', default='tf-idf', choices=['tf-idf', 'count'], help='Vectorizer (feature [count], or fractional [tf-idf] count).')
parser.add_argument('-s', '--size', '--training_set_size', dest='training_set_size', type=int, help='Training set size.')
parser.add_argument('-ng', '--ngram', '--ngram_length', dest='ngram_length', default='unigram', help='N-gram length.', choices=['unigram', 'bigram', 'trigram'])
parser.add_argument('-ne', '--neutral', dest='include_neutral', default=False, help='Include neutral class?', action=argparse.BooleanOptionalAction)
parser.add_argument('-sw', '--stopwords', dest='use_stopwords', default=True, help='Use stop words?', action=argparse.BooleanOptionalAction)

if __name__ == '__main__':
    args = parser.parse_args()

    s = SentimentAnalyzer(args)
    s.train()
    s.test()
