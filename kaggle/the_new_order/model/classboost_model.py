import sys
sys.path.append("../../base")
from common import *
import catboost as cb
from scipy.stats import spearmanr




#----------catboost_model----------

class ClassboostModel:
    def __init__(self, config):
        self.log = logging.getLogger("ClassboostModel")
        self.log.info("model config: {0}".format(config))
        self.config = config
        self.model = cb.CatBoostClassifier(
                #logging_level="Silent",
                loss_function=self.config["loss_function"],
                classes_count=self.config["classes_count"],
                iterations=self.config["iterations"],
                l2_leaf_reg=self.config["l2_leaf_reg"],
                learning_rate=self.config["learning_rate"],
                depth=self.config["depth"],
                thread_count=19,
                one_hot_max_size=self.config["one_hot_max_size"],
                class_weights=[
                    0.008081142895024694,
                    0.018370586437144825,
                    0.13386367905244898,
                    0.3249111486772662,
                    0.5147734429381153
                ]
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
        return prediction.reshape(-1,)
