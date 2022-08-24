import os
import datetime

class ResultsWriter:
    OUTPUT_DIR = 'results'

    def __init__(self, args):
        self._args = args

    def write_result(self, results, number_of_features):
        if not os.path.exists(ResultsWriter.OUTPUT_DIR):
            os.makedirs(ResultsWriter.OUTPUT_DIR)

        train = os.path.basename(os.path.normpath(self._args.training_set))
        test = os.path.basename(os.path.normpath(self._args.test_set))
        alg = self._args.algorithm
        ngram = self._args.ngram_length
        stopwords = 'use-sw' if self._args.use_stopwords else 'no-sw'

        filename = f'./{ResultsWriter.OUTPUT_DIR}/{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{train}_{test}_{alg}_{ngram}_{stopwords}.txt'

        txt_file = open(filename, 'w', encoding='utf-8')
        txt_file.write('=================================\n')
        txt_file.write(f'Arguments: {str(self._args)}\n')
        txt_file.write(f'Number of features: {str(number_of_features)}\n')
        txt_file.write('=================================\n')
        txt_file.write(results)
        txt_file.write('=================================\n')
        txt_file.close() 