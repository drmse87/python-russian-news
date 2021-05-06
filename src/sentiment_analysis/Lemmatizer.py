from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from TokenizerCleaner import TokenizerCleaner

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

class Lemmatizer:
    def __init__(self, training_data = ''):
        self.wnl = WordNetLemmatizer()
        self.tokenizerCleaner = TokenizerCleaner(training_data)

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

        return self.wnl.lemmatize(token, wordnet_pos)

    def __call__(self, doc):
        cleaned_and_tokenized_document = self.tokenizerCleaner.clean_and_tokenize_document(doc)

        return [self.lemmatize_token(token) for token in cleaned_and_tokenized_document]

        

