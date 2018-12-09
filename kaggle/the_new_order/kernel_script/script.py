import json
import warnings
warnings.filterwarnings("ignore")

import logging
import sys

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

        x_known_archive = np.load(self.x_known_path)
        x_known = x_known_archive[x_known_archive.files[0]]

        y_known_archive = np.load(self.y_known_path)
        y_known = y_known_archive[y_known_archive.files[0]]

        known_count = len(x_known)
        known_using_count = int(known_count * self.known_using_part)
        
        self.log.info("known using count: {0}/{1} ({2}%)".format(known_using_count,
            known_count, self.known_using_part * 100))

        random_permutation = np.random.permutation(known_count)
        self.x_known = x_known[random_permutation][:known_using_count]
        self.y_known = y_known[random_permutation][:known_using_count]

        x_to_predict_archive = np.load(self.x_to_predict_path)
        self.x_to_predict = x_to_predict_archive[x_to_predict_archive.files[0]]
        self.log.info("loaded {0} x_to_predict lines".format(len(self.x_to_predict)))

        self.split_known_data_to_train_and_test(config["train_part"])

        self.log.info("inited")

    def split_known_data_to_train_and_test(self, train_part):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_known, self.y_known, train_size = train_part, random_state = 42)
        self.log.info("splitted known data: "
                "x_train size: {0} y_train size: {1} x_test size: {2} y_test size: {3}".format(
                        len(self.x_train), len(self.y_train), len(self.x_test), len(self.y_test)
                )
        )


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

#----------x_transformer_by_config----------

def x_transformer_by_config(config):
    x_transormer_config = config["x_transformer"]
    name = x_transormer_config["name"]
    if (name == "dummy"):
        return DummyXTransformer(x_transormer_config)
    logging.fatal("unknown x transformer name: {0}".format(name))
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
    logging.fatal("unknown model name: {0}".format(name))
#----------config----------
config = json.loads("""
{
  "data_provider": {
    "x_known": "../input/x_train_1.npz",
    "y_known": "../input/y_train.npz",
    "x_to_predict": "../input/x_test.npz",
    "known_using_part" : 0.1,
    "train_part" : 0.9
  },
  "x_transformer": {
    "name": "dummy"
  },
  "model": {
    "name": "catboost",
    "iterations": 1000,
    "depth": 5,
    "loss_function": "MultiClassOneVsAll",
    "classes_count": 5
  },
  "flush_to_file": true,
  "answer_file": "answer.csv"
}
""")
#----------launcher----------

def create_model(config, data_provider):
    x_transformer = x_transformer_by_config(config)
    model = model_by_config(config)

    x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
    model.load_train_data(x_transformer.transform(data_provider.x_train), data_provider.y_train)

    return model

log = logging.getLogger("Launcher")

log.info("launcher config: {0}".format(config))

data_provider = DataProvider(config["data_provider"])

model = create_model(config, data_provider)
#prediction = model.predict(data_provider.x_test)

#print("score:", spearmanr(prediction, data_provider.y_test)[0])

prediction = model.predict(data_provider.x_to_predict)

if (config["flush_to_file"]):
    log.info("flush result to {0}".format(config["answer_file"]))
    answer_file = open(config["answer_file"], 'w')
    answer_file.write("Id,Label\n")

    for i in range(len(prediction)):
        answer_file.write("%s,%s\n" % (i + 1, int(prediction[i][0])))
