import sys
sys.path.append("../../base")
from common import *







#----------catboost_x_transformer----------

class CatboostXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("CatboostXTransformer")
        self.log.info("x_transformer config: {0}".format(config))
        self.config = config
        self.model = cb.CatBoostRegressor(
                #logging_level="Silent",
                loss_function=self.config["loss_function"],
                #classes_count=self.config["classes_count"],
                iterations=self.config["iterations"],
                l2_leaf_reg=self.config["l2_leaf_reg"],
                learning_rate=self.config["learning_rate"],
                depth=self.config["depth"],
                thread_count=19
                #one_hot_max_size=self.config["one_hot_max_size"]
        )
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        additional_column = self.model.predict(x_data)
        print(additional_column.shape)
        print(x_data.shape)
        result = np.concatenate((x_data, additional_column.reshape(-1, 1)), axis=1)
        self.log.info("transformed")
        return result
