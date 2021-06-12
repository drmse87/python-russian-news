import os
import datetime

class ResultsWriter:
    OUTPUT_DIR = 'results'

    def __init__(self, args):
        self._args = args

    def write_result(self, results):
        if not os.path.exists(ResultsWriter.OUTPUT_DIR):
            os.makedirs(ResultsWriter.OUTPUT_DIR)

        train = os.path.basename(os.path.normpath(self._args.training_set))
        test = os.path.basename(os.path.normpath(self._args.test_set))
        alg = self._args.algorithm
        size = self._args.training_set_size if self._args.training_set_size else 'full-size'
        ngram = self._args.ngram_length
        neutral = 'include-neutral' if self._args.include_neutral else 'no-neutral'
        stopwords = 'use-sw' if self._args.use_stopwords else 'no-sw'
        
        filename = f'./{ResultsWriter.OUTPUT_DIR}/{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{train}_{test}_{alg}_{ngram}_{size}_{neutral}_{stopwords}.txt'

        txt_file = open(filename, 'w', encoding='utf-8')
        txt_file.writelines([str(self._args), results])
        txt_file.close() 