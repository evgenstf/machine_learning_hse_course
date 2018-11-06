import sys
sys.path.append("../../base")
from common import *
import string

import nltk
from nltk.stem.porter import PorterStemmer


from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


class TfidfXTransformer:
    def __init__(self, config, need_stemmer):
        self.log = logging.getLogger("TfidfXTransformer")
        self.log.info("x_transformer config:", config)

        self.ngram_range = config["ngram_range"]
        self.min_df = config["min_df"]
        self.max_df = config["max_df"]
        self.max_features = config["max_features"]

        if need_stemmer:
            self.vectorizer = TfidfVectorizer(
                    min_df = self.min_df,
                    max_df = self.max_df,
                    ngram_range = self.ngram_range,
                    lowercase = True,
                    sublinear_tf = True,
                    tokenizer=tokenize,
                    #stop_words = ["must", "and"]
                    #stop_words = stopwords.words('english')
                    norm='l2',
                    max_features = self.max_features
            )
        else:
            self.vectorizer = TfidfVectorizer(
                    min_df = self.min_df,
                    max_df = self.max_df,
                    ngram_range = self.ngram_range,
                    lowercase = True,
                    sublinear_tf = True,
                    #tokenizer=tokenize,
                    #stop_words = ["must", "and"]
                    #stop_words = stopwords.words('english')
                    norm='l2',
                    max_features = self.max_features
            )


        self.name = "tfidf"
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.vectorizer.fit(x_train)
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        result = self.vectorizer.transform(x_data)
        self.log.info("transformed")
        return result

    def features(self):
        return self.vectorizer.get_feature_names()
