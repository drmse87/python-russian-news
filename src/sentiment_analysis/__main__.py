from SentimentAnalyzer import SentimentAnalyzer
import argparse

parser = argparse.ArgumentParser(description='Train a MNB or SVM Sentiment Analyzer.')

parser.add_argument('training_set', metavar='Training set directory', help='Training set directory.')
parser.add_argument('test_set', metavar='Test set directory', help='Test set directory.')
parser.add_argument('-a', '-algorithm', '--algorithm', dest='algorithm', default='MNB', choices=['MNB', 'SVM'], help='Algorithm (MNB or SVM).')
parser.add_argument('-s', '-size', '--training_set_size', dest='size', help='Training set size.')
parser.add_argument('-neutral, --include_neutral', default=True, help='Include neutral class?', action=argparse.BooleanOptionalAction)
parser.add_argument('-ngram, --ngram_length', default='unigram', help='N-gram length.', choices=['unigram', 'bigram', 'trigram'])

if __name__ == '__main__':
    args = parser.parse_args()

    s = SentimentAnalyzer(args)
    s.train()
    s.test()
    
    



