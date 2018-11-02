import sys
sys.path.append("../../base")
from common import *

from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class LogisticRegressionModel:
    def __init__(self, config):
        self.log = logging.getLogger("LogisticRegressionModel")
        self.name = "logistic_regression"
        self.model = LogisticRegression()
        self.log.info("inited")

    def load_train(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(x_train.shape[0], len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")


    def predict_probabilities(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(x_to_predict.shape[0]))
        return self.model.predict_proba(x_to_predict)

    def weights(self):
        return self.model.coef_[0]
