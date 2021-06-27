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
        size = self._args.training_set_size if self._args.training_set_size else 'full-size'
        ngram = self._args.ngram_length
        neutral = 'include-neutral' if self._args.include_neutral else 'no-neutral'
        stopwords = 'use-sw' if self._args.use_stopwords else 'no-sw'

        filename = f'./{ResultsWriter.OUTPUT_DIR}/{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{train}_{test}_{alg}_{ngram}_{size}_{neutral}_{stopwords}.txt'

        txt_file = open(filename, 'w', encoding='utf-8')
        txt_file.write('=================================\n')
        txt_file.write(f'Arguments: {str(self._args)}\n')
        txt_file.write(f'Number of features: {str(number_of_features)}\n')
        txt_file.write('=================================\n')
        txt_file.write(results)
        txt_file.write('=================================\n')
        # if not self._args.training_set_size:
        #     for current_class in most_informative_features:
        #         class_label = current_class[0]
        #         features = current_class[1]
        #         for coef, feat in features:
        #             txt_file.write(f'{class_label} {feat} {coef}\n')
        #         txt_file.write('=================================\n')
        txt_file.close() 