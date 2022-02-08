# Sentiment Analysis of Russian News

## Article Extraction

In the first part of this project, some 395 English language and foreign-facing Russian-language news articles were extracted with the powerful _requests_ and _BeautifulSoup_ libraries.

## Sentiment Analyzer

In the second part of the project, a sentiment analysis was performed on the extracted articles and movie reviews from the Large Movie Review Dataset (https://ai.stanford.edu/~amaas/data/sentiment/) with _sklearn_. The datasets were organized as follows:

### Training set

-   Large Movie Review Dataset (25,000 articles earmarked for training)
-   English news articles (300 articles)

### Test set

-   Large Movie Review Dataset (25,000 articles earmarked for testing)
-   RT and Sputnik articles (95 articles)

### Running the Sentiment Analyzer

Run the command: `python sentiment_analysis` in the src directory with one the following flags:

-   training_set (Training set directory)
-   test_set (Test set directory)
-   -a/--algorithm (Algorithm, valid options are 'MNB' and 'SVM', default: MNB)
-   -v/--vectorizer (Vectorizer, feature 'count', or fractional 'tf-idf' count, default TF-IDF).
-   -s/--size (Training set size).
-   -ng/--ngram (N-gram length, valid options are 'unigram', 'bigram' and 'trigram', default: unigram)
-   -sw/--stopwords or --no-stopwords (Use stopwords, default: True)
