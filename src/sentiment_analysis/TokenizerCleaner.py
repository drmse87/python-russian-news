import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import strip_accents_unicode

# nltk.download('stopwords')
# string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

class TokenizerCleaner:
    def __init__(self, args):
        self.args = args
        self.extended_stop_words = stopwords.words('english')
        
        # Extend the stop words.
        self.extended_stop_words.append("n't")
        self.extended_stop_words.append("didn't")

        # Add specific stop words for when training with movie reviews.
        if ('imdb' in args.training_set):
            self.extended_stop_words.append('actor')
            self.extended_stop_words.append('actress')
            self.extended_stop_words.append('director')
            self.extended_stop_words.append('scene')
            self.extended_stop_words.append('movie')
            self.extended_stop_words.append('film')

    def clean_document(self, doc):       
        return strip_accents_unicode(doc) \
                  .replace('<br />', ' ') \
                  .translate(str.maketrans('', '', string.digits)) \
                  .strip(string.punctuation)

    def tokenize_document(self, doc):
        if not self.args.use_stopwords:
            return [token for token in word_tokenize(doc)]
        
        return [
            token for token in word_tokenize(doc) 
                if token not in self.extended_stop_words
                    ]

    def clean_and_tokenize_document(self, doc):
        cleaned_document = self.clean_document(doc)

        return self.tokenize_document(cleaned_document)
    