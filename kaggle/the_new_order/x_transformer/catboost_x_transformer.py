import sys
sys.path.append("../../base")
from common import *







#----------catboost_x_transformer----------

class CatboostXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("CatboostXTransformer")
        self.log.info("x_transformer config: {0}".format(config))
        self.config = config
        primary_config = config["primary_model"]
        self.model = cb.CatBoostRegressor(
                #logging_level="Silent",
                loss_function=primary_config["loss_function"],
                #classes_count=self.config["classes_count"],
                iterations=primary_config["iterations"],
                l2_leaf_reg=primary_config["l2_leaf_reg"],
                learning_rate=primary_config["learning_rate"],
                #bagging_temperature=self.config["bagging_temperature"],
                depth=primary_config["depth"],
                thread_count=19,
                random_state=42
                #one_hot_max_size=self.config["one_hot_max_size"]
        )
        secondary_config = config["secondary_model"]
        self.secondary_model = cb.CatBoostClassifier(
                #logging_level="Silent",
                loss_function="MultiClassOneVsAll",
                classes_count=secondary_config["classes_count"],
                iterations=secondary_config["iterations"],
                l2_leaf_reg=secondary_config["l2_leaf_reg"],
                learning_rate=secondary_config["learning_rate"],
                #bagging_temperature=self.config["bagging_temperature"],
                depth=secondary_config["depth"],
                thread_count=19,
                random_state=42
                #one_hot_max_size=self.config["one_hot_max_size"]
        )
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.log.info("start primary model")
        self.model.fit(x_train, y_train)

        self.log.info("start secondary model")
        self.secondary_model.fit(x_train, y_train)
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        prediction = self.model.predict(x_data).reshape(-1, 1)
        result = np.concatenate((x_data, prediction), axis=1)
        result = np.concatenate((result, np.log(prediction)), axis=1)
        result = np.concatenate((result, np.sqrt(prediction)), axis=1)
        result = np.concatenate((result, prediction ** 2), axis=1)

        self.log.info("transform x_data size: {0}".format(len(x_data)))
        secondary_prediction = self.secondary_model.predict_proba(x_data)
        result = np.concatenate((result, secondary_prediction), axis=1)
        result = np.concatenate((result, np.log(secondary_prediction)), axis=1)
        result = np.concatenate((result, np.sqrt(secondary_prediction)), axis=1)
        result = np.concatenate((result, secondary_prediction ** 2), axis=1)

        """
        additional_column = prediction ** 2
        result = np.concatenate((result, additional_column.reshape(-1, 1)), axis=1)

        additional_column = prediction ** 3
        result = np.concatenate((result, additional_column.reshape(-1, 1)), axis=1)

        additional_column = prediction ** 0.5
        result = np.concatenate((result, additional_column.reshape(-1, 1)), axis=1)

        additional_column = prediction ** 0.25
        result = np.concatenate((result, additional_column.reshape(-1, 1)), axis=1)
        """


        self.log.info("transformed")
        return result
