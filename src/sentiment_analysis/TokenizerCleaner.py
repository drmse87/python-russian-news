import string
import regex
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import strip_accents_unicode

# nltk.download('stopwords')
# string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

class TokenizerCleaner:
    def __init__(self, args):
        self._args = args
        self._extended_stop_words = stopwords.words('english')

        # string.punctuation does not include these chars...
        self._additional_unwanted_chars = ['“', '”', '–', '©', '’', '×']

        # Extend the stop words.
        self._extended_stop_words.append("n't")
        self._extended_stop_words.append("didn't")

        # Add specific stop words for when training with movie reviews.
        if ('imdb' in args.training_set):
            self._extended_stop_words.append('actor')
            self._extended_stop_words.append('actress')
            self._extended_stop_words.append('director')
            self._extended_stop_words.append('scene')
            self._extended_stop_words.append('movie')
            self._extended_stop_words.append('film')

    def clean_document(self, doc):
        # Remove HTML.
        TAG_RE = regex.compile(r'<[^>]+>')
        mystr = TAG_RE.sub('', doc)

        # Remove digits and punctuation.              
        mystr = strip_accents_unicode(doc) \
                  .translate(str.maketrans('', '', string.digits)) \
                  .translate(str.maketrans('', '', string.punctuation))

        # Remove additional unwanted chars.
        for unwanted_char in self._additional_unwanted_chars:
            if unwanted_char in doc:
                mystr = mystr.replace(unwanted_char, ' ') \

        return mystr

    def tokenize_document(self, doc):
        if not self._args.use_stopwords:
            return [token for token in word_tokenize(doc)]
        
        return [token for token in word_tokenize(doc) if token not in self._extended_stop_words]

    def clean_and_tokenize_document(self, doc):
        cleaned_document = self.clean_document(doc)

        return self.tokenize_document(cleaned_document)
    