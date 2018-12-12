import json
import warnings
warnings.filterwarnings("ignore")

import logging
import sys

import random

from scipy.stats import spearmanr

import catboost as cb

logging.basicConfig(level=logging.INFO)

#root = logging.getLogger()
#root.setLevel(logging.DEBUG)
#ch = logging.StreamHandler(sys.stdout)
#ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s %(levelname)s[%(name)s] - %(message)s')
#ch.setFormatter(formatter)
#root.addHandler(ch)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from math import sqrt



def draw_pair_plot(x_data, y_data):
    ncol, nrow = 7, x_data.shape[1] // 7 + (x_data.shape[1] % 7 > 0)
    plt.figure(figsize=(ncol * 4, nrow * 4))

    for i, feature in enumerate(x_data.columns):
        plt.subplot(nrow, ncol, i + 1)
        plt.scatter(x_data[feature], y_data, s=10, marker='o', alpha=.6)
        plt.xlabel(feature)
        if i % ncol == 0:
            plt.ylabel('target')

def mape_score(y_data, prediction): 
    total = 0
    bad_cnt = 0
    for i in range(len(y_data.as_matrix())):
        loss_value = np.abs((y_data.as_matrix()[i][0] - prediction[i]) / (y_data.as_matrix()[i][0]))
        if (loss_value > 0.08):
            print("loss_value:", loss_value, "index:", i)
            bad_cnt += 1
        total += loss_value
    print("bad_cnt:", bad_cnt)
    total /= len(y_data)
    total = total * 100
    return total

def ratio_score(y_expected, y_predicted):
    return roc_auc_score(y_expected[:len(y_predicted)], y_predicted)

#---------data_provider----------

class DataProvider:
    def __init__(self, config):
        self.log = logging.getLogger("DataProvider")
        self.log.info("data provider config: {0}".format(config))
        self.x_known_path = config["x_known"]
        self.y_known_path = config["y_known"]
        self.x_to_predict_path = config["x_to_predict"]
        self.known_using_part = config["known_using_part"]
        x_known = None

        for i in range(1, config["max_file_index"] + 1):
            filename = self.x_known_path.replace("{i}", str(i))
            with np.load(filename) as data:
                self.log.info("load {0}".format(filename))
                temp_data = data[data.files[0]]
                if x_known is None:
                    x_known = temp_data
                else:
                    x_known = np.concatenate((x_known, temp_data))

        y_known = None
        with np.load(self.y_known_path) as data:
            self.log.info("load {0}".format(self.y_known_path))
            y_known = data[data.files[0]]


        known_count = len(x_known)
        known_using_count = int(known_count * self.known_using_part)

        self.log.info("known using count: {0}/{1} ({2}%)".format(known_using_count,
            known_count, self.known_using_part * 100))

        np.random.seed(seed=0)
        random_permutation = np.random.permutation(known_count)
        self.x_known = x_known[random_permutation][:known_using_count]
        self.y_known = y_known[random_permutation][:known_using_count]

        with np.load(self.x_to_predict_path) as data:
            self.log.info("load {0}".format(self.x_to_predict_path))
            self.x_to_predict = data[data.files[0]]

        """
        for i in config["features_to_multiply"]:
            self.x_known = np.concatenate((self.x_known, np.log(self.x_known[:, i]).reshape(-1, 1)), axis=1)
            self.x_known = np.concatenate((self.x_known, np.sqrt(self.x_known[:, i]).reshape(-1, 1)), axis=1)
            self.x_known = np.concatenate((self.x_known, (self.x_known[:, i] ** 2).reshape(-1, 1)), axis=1)

            self.x_to_predict = np.concatenate((self.x_to_predict, np.log(self.x_to_predict[:, i]).reshape(-1, 1)), axis=1)
            self.x_to_predict = np.concatenate((self.x_to_predict, np.sqrt(self.x_to_predict[:, i]).reshape(-1, 1)), axis=1)
            self.x_to_predict = np.concatenate((self.x_to_predict, (self.x_to_predict[:, i] ** 2).reshape(-1, 1)), axis=1)
            """


        """
        for i in config["features_to_multiply"]:
            for j in config["features_to_multiply"]:
                if i == j:
                    continue
                self.x_known = np.concatenate((self.x_known, (np.log(self.x_known[:, i]) * np.log(self.x_known[:, j])).reshape(-1, 1)), axis=1)

                self.x_to_predict = np.concatenate((self.x_to_predict, (np.log(self.x_to_predict[:, i]) * np.log(self.x_to_predict[:, j])).reshape(-1, 1)), axis=1)

        self.log.info("loaded x_known rows: {0} columns: {1}".format(self.x_known.shape[0], self.x_known.shape[1]))
        self.log.info("loaded x_to_predict rows: {0} columns: {1}".format(self.x_to_predict.shape[0], self.x_to_predict.shape[1]))

        """
        for i in config["features_to_throw"]:
            self.x_known = np.delete(self.x_known, i, 1)
            self.x_to_predict = np.delete(self.x_to_predict, i, 1)

        self.split_known_data_to_train_and_test(config["train_part"])

        del self.x_known
        del self.y_known

        self.log.info("inited")

    def split_known_data_to_train_and_test(self, train_part):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_known, self.y_known, train_size = train_part, random_state = 42)
        self.log.info("splitted known data: "
                "x_train size: {0} y_train size: {1} x_test size: {2} y_test size: {3}".format(
                    len(self.x_train), len(self.y_train), len(self.x_test), len(self.y_test)
                    )
                )
        """
        bx, by = self.balance_data(self.x_train, self.y_train)
        self.x_train = bx
        self.y_train = by
        """

    def balance_data(self, x, y):
        indixes = [[], [], [], [], []]

        for i in range(1, len(x)):
            indixes[y[i]].append(i)

        fsize = len(indixes[4])

        def sample(cls):
            ind = [indixes[cls][i] for i in sorted(random.sample(range(len(indixes[cls])), fsize))]
            return x[ind], y[ind]

        xs, ys = sample(0)
        for i in range(1, 5):
            xx, yy = sample(i)
            xs = np.concatenate((xs, xx), axis=0)
            ys = np.concatenate((ys, yy), axis=0)


        perm = np.random.permutation(xs.shape[0])
        x_train = xs[perm]
        y_train = ys[perm]
        return x_train, y_train


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

#----------dummy_x_transformer----------

class DummyXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("DummyXTransformer")
        self.log.info("x_transformer config: {0}".format(config))
        self.config = config
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.x_train = x_train
        self.y_train = y_train
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        result = x_data
        self.log.info("transformed")
        return result

#----------regboost_x_transformer----------

class RegboostXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("RegboostXTransformer")
        self.log.info("x_transformer config: {0}".format(config))
        self.config = config
        self.model = cb.CatBoostRegressor(
                #logging_level="Silent",
                loss_function=config["loss_function"],
                #classes_count=self.config["classes_count"],
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
        prediction = self.model.predict(x_data).reshape(-1, 1)
        x_data = np.concatenate((x_data, prediction), axis=1)
        x_data = np.concatenate((x_data, np.log(prediction)), axis=1)
        #result = np.concatenate((result, np.sqrt(prediction)), axis=1)
        #result = np.concatenate((result, prediction ** 2), axis=1)
        self.log.info("transformed")
        return x_data
#----------dummy_x_transformer----------

class SuperXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("SuperXTransformer")
        self.log.info("x_transformer config: {0}".format(config))
        self.config = config
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.x_train = x_train
        self.y_train = y_train
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        result = x_data
        self.log.info("transformed")
        return result
#----------x_transformer_by_config----------

def x_transformer_by_config(config):
    name = config["name"]
    if (name == "dummy"):
        return DummyXTransformer(config)
    if (name == "super"):
        return SuperXTransformer(config)
    if (name == "catboost"):
        return CatboostXTransformer(config)
    if (name == "regboost"):
        return RegboostXTransformer(config)
    logging.fatal("unknown x transformer name: {0}".format(name))
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
#----------dummy_model----------

class DummyModel:
    def __init__(self, config):
        self.log = logging.getLogger("DummyModel")
        self.log.info("model config: {0}".format(config))
        self.config = config
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.x_train = x_train
        self.y_train = y_train
        self.log.info("loaded")

    def predict(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(len(x_to_predict)))
        result = [1]
        self.log.info("predicted")
        return result
#----------model_by_config----------

def model_by_config(config):
    model_config = config["model"]
    name = model_config["name"]
    if (name == "dummy"):
        return DummyModel(model_config)
    if (name == "catboost"):
        return CatboostModel(model_config)
    if (name == "regboost"):
        return RegboostModel(model_config)
    logging.fatal("unknown model name: {0}".format(name))
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
                metric_period=10,
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
#----------config----------
config = json.loads("""
{
  "data_provider": {
    "x_known": "../input/x_train_{i}.npz",
    "y_known": "../input/y_train.npz",
    "x_to_predict": "../input/x_test.npz",
    "max_file_index": 1,
    "known_using_part" : 1,
    "train_part" : 0.9,
    "features_to_multiply": [16, 47, 69, 70, 102, 135],
    "features_to_throw": [9, 18, 90, 103, 109, 122]
  },
  "primary_x_transformer": {
    "name": "dummy",
    "iterations": 10,
    "depth": 10,
    "learning_rate": 0.3,
    "l2_leaf_reg":0.07,
    "loss_function": "RMSE"
  },
  "secondary_x_transformer": {
    "name": "dummy",
    "iterations": 10,
    "depth": 10,
    "learning_rate": 0.3,
    "l2_leaf_reg":0.07,
    "loss_function": "MultiClassOneVsAll",
    "classes_count": 5
  },
  "model": {
    "name": "regboost",
    "iterations": 5,
    "depth": 8,
    "learning_rate": 0.1,
    "l2_leaf_reg":0.7,
    "loss_function": "RMSE",
    "classes_count": 5
  },
  "predict_answer": true,
  "answer_file": "answer.csv",

  "features_to_throw": [9, 18, 90, 103, 109, 122],
  "using_features": [3, 5, 6, 7, 13, 16, 19, 23, 29, 31, 34, 38, 40, 44, 53, 56, 57, 69, 70, 72, 76, 88, 89, 95, 101, 102, 106, 127, 130, 131, 146]
}
""")
#----------launcher----------

log = logging.getLogger("Launcher")

log.info("launcher config: {0}".format(config))

data_provider = DataProvider(config["data_provider"])

primary_x_transformer = x_transformer_by_config(config["primary_x_transformer"])
primary_x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)

x_train_transformed = primary_x_transformer.transform(data_provider.x_train)
del data_provider.x_train
x_test_transformed = primary_x_transformer.transform(data_provider.x_test)
del data_provider.x_test
x_to_predict_transformed = primary_x_transformer.transform(data_provider.x_to_predict)
del data_provider.x_to_predict

del primary_x_transformer


secondary_x_transformer = x_transformer_by_config(config["secondary_x_transformer"])
secondary_x_transformer.load_train_data(x_train_transformed, data_provider.y_train)

x_train_transformed = secondary_x_transformer.transform(x_train_transformed)
x_test_transformed = secondary_x_transformer.transform(x_test_transformed)
x_to_predict_transformed = secondary_x_transformer.transform(x_to_predict_transformed)

del secondary_x_transformer

model = model_by_config(config)
model.load_train_data(
        x_train_transformed,
        data_provider.y_train
)

del x_train_transformed
del data_provider.y_train

prediction = model.predict(x_test_transformed)

#print("prediction:", prediction)
score = spearmanr(prediction, data_provider.y_test)[0]
print("score:", score)
score_file = open("%s" % score, 'w')
score_file.write("%s" % score)


if (config["predict_answer"]):
    prediction = model.predict(x_to_predict_transformed)
    log.info("flush result to {0}".format(config["answer_file"]))
    answer_file = open(config["answer_file"], 'w')
    answer_file.write("Id,Label\n")

    for i in range(len(prediction)):
        answer_file.write("%s,%s\n" % (i + 1, int(prediction[i])))
