import sys
sys.path.append("../../base")
from common import *
import catboost as cb
from scipy.stats import spearmanr




#----------catboost_model----------

class CatboostModel:
    def __init__(self, config):
        self.log = logging.getLogger("CatboostModel")
        self.log.info("model config: {0}".format(config))
        self.config = config
        self.model = cb.CatBoostClassifier(
                #logging_level="Silent",
                loss_function=self.config["loss_function"],
                classes_count=self.config["classes_count"],
                iterations=self.config["iterations"],
                depth=self.config["depth"],
                thread_count=19
        )
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")

    def predict(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(len(x_to_predict)))
        prediction = self.model.predict(x_to_predict)
        self.log.info("predicted")
        return prediction
