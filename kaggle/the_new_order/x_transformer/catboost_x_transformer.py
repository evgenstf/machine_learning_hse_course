import sys
sys.path.append("../../base")
from common import *







#----------catboost_x_transformer----------

class CatboostXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("CatboostXTransformer")
        self.log.info("x_transformer config: {0}".format(config))
        self.config = config
        self.model = cb.CatBoostClassifier(
                #logging_level="Silent",
                loss_function="MultiClassOneVsAll",
                classes_count=config["classes_count"],
                iterations=config["iterations"],
                l2_leaf_reg=config["l2_leaf_reg"],
                learning_rate=config["learning_rate"],
                #bagging_temperature=self.config["bagging_temperature"],
                depth=config["depth"],
                thread_count=19,
                metric_period=10,
                random_state=42
                #one_hot_max_size=self.config["one_hot_max_size"]
        )
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        prediction = self.model.predict_proba(x_data)
        x_data = np.concatenate((x_data, prediction), axis=1)
        x_data = np.concatenate((x_data, np.log(prediction)), axis=1)
        x_data = np.concatenate((x_data, np.sqrt(prediction)), axis=1)
        x_data = np.concatenate((x_data, prediction ** 2), axis=1)
        self.log.info("transformed")
        return x_data
