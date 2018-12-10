import sys
sys.path.append("../../base")
from common import *
import catboost as cb
from scipy.stats import spearmanr




#----------catboost_model----------

class RegboostModel:
    def __init__(self, config):
        self.log = logging.getLogger("RegboostModel")
        self.log.info("model config: {0}".format(config))
        self.config = config
        self.model = cb.CatBoostRegressor(
                #logging_level="Silent",
                loss_function=self.config["loss_function"],
                #classes_count=self.config["classes_count"],
                iterations=self.config["iterations"],
                l2_leaf_reg=self.config["l2_leaf_reg"],
                learning_rate=self.config["learning_rate"],
                depth=self.config["depth"],
                #bagging_temperature=self.config["bagging_temperature"],
                thread_count=19,
                random_state=42
                #one_hot_max_size=self.config["one_hot_max_size"]
        )
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.model.fit(x_train, y_train)
        """
        self.log.info("loaded")
        for i in range(150):
            print(i, self.model.feature_importances_[i])
        exit()
        """
        self.log.info("loaded")

    def predict(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(len(x_to_predict)))
        prediction = self.round_prediction(self.model.predict(x_to_predict))
        #prediction = self.model.predict(x_to_predict)
        self.log.info("predicted")
        return prediction

    def round_prediction(self, prediction):
        result = [0] * len(prediction)
        sort_ind = np.argsort(prediction)

        for i in range(len(sort_ind)):
            result[sort_ind[i]] = int(i / len(sort_ind) * 21)
        return result

"""


        percentiles = np.cumsum()
        print(percentiles)
        result = np.empty_like(prediction)
        result[prediction < np.percentile(prediction, percentiles[0])] = 0
        for i in range(15):
            result[
                    np.logical_and(prediction >= np.percentile(prediction, percentiles[i]),
                    prediction < np.percentile(prediction, percentiles[i + 1]))
            ] = i + 1
        result[prediction >= np.percentile(prediction, percentiles[15])] = 15
        return result
"""
"""
    def round_prediction(self, prediction):
        percentiles = np.cumsum(np.array([
            51.477 / 5,
            51.477 / 5,
            51.477 / 5,
            51.477 / 5,
            51.477 / 5,
            32.491 / 5,
            32.491 / 5,
            32.491 / 5,
            32.491 / 5,
            32.491 / 5,
            13.386 / 5,
            13.386 / 5,
            13.386 / 5,
            13.386 / 5,
            13.386 / 5,
            1.837 / 5,
            1.837 / 5,
            1.837 / 5,
            1.837 / 5
        ]))
        result = np.empty_like(prediction)
        result[prediction < np.percentile(prediction, percentiles[0])] = 0
        for i in range(18):
            result[
                    np.logical_and(prediction >= np.percentile(prediction, percentiles[i]),
                    prediction < np.percentile(prediction, percentiles[i + 1]))
            ] = i + 1
        result[prediction >= np.percentile(prediction, percentiles[18])] = 19
        return result
        """
