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
        catboost_parameters = {
                'iterations': config["iterations"],
                'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=5'],
                'random_seed': 0,
                'loss_function': config["loss_function"]
        }

        self.model = cb.CatBoost(catboost_parameters)
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")

    def predict(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(len(x_to_predict)))
        prediction = self.round_prediction(self.model.predict(x_to_predict))
        self.log.info("predicted")
        return prediction.reshape(-1,)

    def round_prediction(self, prediction):
        percentiles = np.cumsum(np.array([25.7385, 25.7385, 16.245, 16.245, 6.5, 6.5, 0.9,
            0.9]))
        print(percentiles)
        result = np.empty_like(prediction)
        result[prediction < np.percentile(prediction, percentiles[0])] = 0
        for i in range(7):
            result[
                    np.logical_and(prediction >= np.percentile(prediction, percentiles[i]),
                    prediction < np.percentile(prediction, percentiles[i + 1]))
            ] = i + 1
        result[prediction >= np.percentile(prediction, percentiles[7])] = 7
        return result
