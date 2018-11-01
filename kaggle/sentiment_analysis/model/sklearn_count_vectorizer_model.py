import sys
sys.path.append("../../base")
from common import *

from sklearn.feature_extraction import text

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
        X_train_counts = self.count_vectorizer.fit_transform(["the a bad lel", "a the good kek"])
        tfidf_transformer = text.TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        return X_train_tfidf

    def predict(self, x_to_predict):
        return 1


