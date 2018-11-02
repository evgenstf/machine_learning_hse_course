import sys
sys.path.append("../../base")
from common import *

from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class SklearnCountVectorizerModel:
    def __init__(self, config):
        self.log = logging.getLogger("SkLearnCountVectorizerModel")
        self.name = "sklearn_count_vectorizer"
        self.max_features_count = config["max_features_count"]
        self.ngram_range = (config["ngram_range"][0], config["ngram_range"][1])
        self.count_vectorizer = text.CountVectorizer(
                ngram_range = self.ngram_range,
                max_features = self.max_features_count
        )
        self.log.info("inited")

    def load_train(self, x_train, y_train):
        print("size:", len(x_train), "tests:", x_train)
        print(y_train)
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(x_train)
        self.tfidf = self.vectorizer.transform(x_train)

        self.lr = LogisticRegression()
        self.lr.fit(self.tfidf, y_train)


    def predict(self, x_to_predict):
        return self.lr.predict_proba(self.vectorizer.transform(x_to_predict))


