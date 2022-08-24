import string
import regex as re
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import strip_accents_unicode

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

class Lemmatizer:
    def __init__(self, args):
        self._wnl = WordNetLemmatizer()
        self._tokenizerCleaner = TokenizerCleaner(args)

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_token(self, token):
        token_and_pos_tag = pos_tag([token])
        token_pos_tag = token_and_pos_tag[0][1]
        wordnet_pos = self.get_wordnet_pos(token_pos_tag)

        return self._wnl.lemmatize(token, wordnet_pos)

    def __call__(self, doc):
        cleaned_and_tokenized_document = self._tokenizerCleaner.clean_and_tokenize_document(doc)

        return [self.lemmatize_token(token) for token in cleaned_and_tokenized_document]

class TokenizerCleaner:
    def __init__(self, args):
        self._args = args
        self._extended_stop_words = stopwords.words('english')

        # string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ (does not include these chars)
        self._additional_unwanted_chars = ['“', '”', '—', '–', '©', '’']

        # Extend the stop words.
        self._extended_stop_words.append("n't")
        self._extended_stop_words.append("didn't")

    def clean_document(self, doc):
        # Remove HTML.
        cleaned_doc = re.sub('<[^<]+?>', ' ', doc)

        # Remove Unicode chars.
        cleaned_doc = cleaned_doc.encode('ascii', 'ignore')
        cleaned_doc = cleaned_doc.decode()

        # Remove digits and punctuation.
        cleaned_doc = strip_accents_unicode(cleaned_doc) \
                  .translate(str.maketrans(' ', ' ', string.digits)) \
                  .translate(str.maketrans(' ', ' ', string.punctuation))

        # Remove additional unwanted chars.
        for unwanted_char in self._additional_unwanted_chars:
            if unwanted_char in cleaned_doc:
                cleaned_doc = cleaned_doc.replace(unwanted_char, ' ') \

        return cleaned_doc

    def tokenize_document(self, doc):
        if not self._args.use_stopwords:
            return [token for token in word_tokenize(doc)]
        
        return [token for token in word_tokenize(doc) if token not in self._extended_stop_words]

    def clean_and_tokenize_document(self, doc):
        cleaned_document = self.clean_document(doc)

        return self.tokenize_document(cleaned_document)
    