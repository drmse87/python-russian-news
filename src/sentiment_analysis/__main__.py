from SentimentAnalyzer import SentimentAnalyzer
import argparse

parser = argparse.ArgumentParser(description='Train a MNB or SVM Sentiment Analyzer.')

parser.add_argument('training_set', metavar='Training set directory', help='Training set directory.')
parser.add_argument('test_set', metavar='Test set directory', help='Test set directory.')
parser.add_argument('-a', '-algorithm', '--algorithm', dest='algorithm', default='MNB', choices=['MNB', 'SVM'], help='Algorithm.')
parser.add_argument('-s', '-size', '--training_set_size', dest='size', type=int, help='Training set size.')
parser.add_argument('-ngram, --ngram_length', dest='ngram', default='unigram', help='N-gram length.', choices=['unigram', 'bigram', 'trigram'])
parser.add_argument('-neutral, --include_neutral', dest='include_neutral', default=False, help='Include neutral class?', action=argparse.BooleanOptionalAction)
parser.add_argument('-stopwords, --use_stopwords', dest='use_stopwords', default=True, help='Use stopwords?', action=argparse.BooleanOptionalAction)

if __name__ == '__main__':
    args = parser.parse_args()

    s = SentimentAnalyzer(args)
    s.train()
    s.test()
    
    



