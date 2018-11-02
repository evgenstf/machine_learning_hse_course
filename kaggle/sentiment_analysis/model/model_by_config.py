import sys
sys.path.append("../../base")
from common import *

from dummy_model import *
from word_weight_model import *
from sklearn_count_vectorizer_model import *
from logistic_regression_model import *

def model_by_config(config):
    model_config = config["model"]
    name = model_config["name"]
    if (name == "dummy"):
        return DummyModel(model_config)
    if (name == "word_weight"):
        return WordWeightModel(model_config)
    if (name == "sklearn_count_vectorizer"):
        return SklearnCountVectorizerModel(model_config)
    if (name == "logistic_regression"):
        return LogisticRegressionModel(model_config)
    logging.fatal("unknown model name: {0}".format(name))
