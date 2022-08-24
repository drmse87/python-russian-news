# Sentiment Analysis of Russian News

## Article Extraction
In the first part of this project, some 395 English language and foreign-facing Russian-language news articles were extracted with the powerful _requests_ and _BeautifulSoup_ libraries.

## Sentiment Analyzer
In the second part of the project, a sentiment analysis was performed on the extracted articles along with movie reviews from the Large Movie Review Dataset (https://ai.stanford.edu/~amaas/data/sentiment/) with _sklearn_. 

The datasets were organized as follows:

### Training set
* Large Movie Review Dataset (25,000 articles earmarked for training)
* English news articles (300 articles)

### Test set
* Large Movie Review Dataset (25,000 articles earmarked for testing)
* RT and Sputnik articles (95 articles)

### Running the Sentiment Analyzer
Although the Sentiment Analyzer was developed for specific datasets, it has since been made generic, for example, by using dynamic instead of hard coded labels, not adding domain specific stop words and so forth. Dataset specific flags and flags related to benchmarking have also been removed.

The Sentiment Analyzer can be run with the command: `python .src/sentiment_analysis {training_set_dir} {test_set_dir}` with either of the optional flags:

* -a/--algorithm (Algorithm, valid options are 'MNB' and 'SVM', default: MNB)
* -v/--vectorizer (Vectorizer, feature 'count', or fractional 'tf-idf' count, default TF-IDF).
* -ng/--ngram (N-gram length, valid options are 'unigram', 'bigram' and 'trigram', default: unigram)
* -sw/--stopwords or --no-stopwords (Use stopwords, default: True)

The script checks the provided training and test set paths for appropriate labels (subdirectories) and documents (any .txt files inside them). Note that test and training set directories should be organized as follows (else the script should raise an error):
* Training set
    * Label 1
        * File 1.txt
        * File 2.txt
        * ...
    * Label 2
        * File 1.txt
        * File 2.txt
        * ...
* Test set
    * Label 1
        * File 1.txt
        * File 2.txt
        * ...
    * Label 2
        * File 1.txt
        * File 2.txt
        * ...

Finally it presents the result in a classification report.