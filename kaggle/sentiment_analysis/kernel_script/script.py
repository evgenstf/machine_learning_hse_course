import json
import warnings
warnings.filterwarnings("ignore")

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s[%(name)s] - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
import sys
print("../base")

from sklearn.model_selection import train_test_split
from random import shuffle
import pandas as pd
import langdetect

"""
try:
    if langdetect.detect(self.x_known[i]) != "en":
        self.x_known[i] = ""
    else:
        self.log.debug("not english: {0}".format(self.x_known[i]))
except Exception:
    self.log.error("could not detect language: {0}".format(self.x_known[i]))
"""

class DataProvider:
    def __init__(self, config, part):
        self.log = logging.getLogger("DataProvider")
        self.log.info("data provider config: {0}".format(config))
        self.x_known_path = config["x_known"]
        self.y_known_path = config["y_known"]
        self.x_to_predict_path = config["x_to_predict"]
        self.known_using_part = config["known_using_part"]

        x_known_file = open(self.x_known_path, 'r')
        self.x_known = x_known_file.readlines()
        self.y_known = np.array(pd.read_csv(self.y_known_path)['Probability'].values)

        known_using_count = int(len(self.x_known) * self.known_using_part)
        self.x_known = self.x_known[:known_using_count]
        self.y_known = self.y_known[:known_using_count]

        self.log.debug("loaded {0} x_known lines".format(len(self.x_known)))
        self.log.debug("loaded {0} y_known lines".format(len(self.y_known)))

        x_to_predict_file = open(self.x_to_predict_path, 'r')
        self.x_to_predict = x_to_predict_file.readlines()
        self.log.debug("loaded {0} x_to_predict lines".format(len(self.x_to_predict)))

        for i in range(len(self.x_known)):
            self.x_known[i] = self.x_known[i].split(':', 1)[0 if part == "header" else 1]

        for i in range(len(self.x_to_predict)):
            self.x_to_predict[i] = self.x_to_predict[i].split(':', 1)[0 if part == "header" else 1]

        self.split_known_data_to_train_and_test(config["train_part"])

        self.log.info("inited")

    def split_known_data_to_train_and_test(self, train_part):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_known, self.y_known, train_size = train_part, random_state = 42)
        self.log.debug("splitted known data: "
                "x_train size: {0} y_train size: {1} x_test size: {2} y_test size: {3}".format(
                        len(self.x_train), len(self.y_train), len(self.x_test), len(self.y_test)
                )
        )


import sys
print("../../base")


class DummyXTransformer:
    def __init__(self, config):
        self.name = "dummy"
        self.config = config
import sys
print("../../base")

import string

import nltk
from nltk.stem.porter import PorterStemmer


from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


class TfidfXTransformer:
    def __init__(self, config, need_stemmer):
        self.log = logging.getLogger("TfidfXTransformer")
        self.log.info("x_transformer config:", config)

        self.ngram_range = config["ngram_range"]
        self.min_df = config["min_df"]
        self.max_df = config["max_df"]
        self.max_features = config["max_features"]

        if need_stemmer:
            self.vectorizer = TfidfVectorizer(
                    min_df = self.min_df,
                    max_df = self.max_df,
                    ngram_range = self.ngram_range,
                    lowercase = True,
                    sublinear_tf = True,
                    tokenizer=tokenize,
                    #stop_words = ["must", "and"]
                    #stop_words = stopwords.words('english')
                    norm='l2',
                    max_features = self.max_features
            )
        else:
            self.vectorizer = TfidfVectorizer(
                    min_df = self.min_df,
                    max_df = self.max_df,
                    ngram_range = self.ngram_range,
                    lowercase = True,
                    sublinear_tf = True,
                    #tokenizer=tokenize,
                    #stop_words = ["must", "and"]
                    #stop_words = stopwords.words('english')
                    norm='l2',
                    max_features = self.max_features
            )


        self.name = "tfidf"
        self.log.info("inited")

    def load_train_data(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(len(x_train), len(y_train)))
        self.vectorizer.fit(x_train)
        self.log.info("loaded")

    def transform(self, x_data):
        self.log.info("transform x_data size: {0}".format(len(x_data)))
        result = self.vectorizer.transform(x_data)
        self.log.info("transformed")
        return result

    def features(self):
        return self.vectorizer.get_feature_names()
import sys
print("../../base")

import string

class WordCounterXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("WordCounterXTransformer")
        self.log.info("x_transformer config:", config)

        self.name = "word_counter"
        self.min_word_len = config["min_word_len"]
        self.log.info("inited")

    def transform(self, x_data):
        transformed_data = []
        for i in range(len(x_data)):
            """
            separator_index = x_data[i].find(':')
            line = x_data[i][:separator_index]
            """
            line = x_data[i].lower()
            counts = {}
            previous_word = ""
            for word in [word.strip(string.punctuation) for word in line.split()]:
                if (len(word) >= self.min_word_len):
                        if previous_word + word not in counts:
                            counts[previous_word + word] = 0
                        counts[previous_word + word] += 1
                previous_word = word
            transformed_data.append(counts)
            if (i % 10000 == 0):
                self.log.debug("transformed line: {0}/{1}".format(i, len(x_data)))
        return transformed_data
import sys
print("../../base")






def x_transformer_by_config(config, need_stemmer):
    x_transormer_config = config["x_transformer"]
    name = x_transormer_config["name"]
    if (name == "dummy"):
        return DummyXTransformer(x_transormer_config)
    if (name == "word_counter"):
        return WordCounterXTransformer(x_transormer_config)
    if (name == "tfidf"):
        return TfidfXTransformer(x_transormer_config, need_stemmer)
    logging.fatal("unknown x transformer name: {0}".format(name))
class DummyModel:
    def __init__(self, config):
        self.name = "dummy"
        self.config = config
import sys
print("../../base")


from sklearn.feature_extraction import text
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

class LinearSVCModel:
    def __init__(self, config):
        self.log = logging.getLogger("LinearSVCModel")
        self.name = "linear_svc"
        self.model = LinearSVC()
        self.log.info("inited")

    def load_train(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(x_train.shape[0], len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")


    def predict_probabilities(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(x_to_predict.shape[0]))
        predictions = 1 /(1 + np.exp(self.model.decision_function(-x_to_predict)))
        probabilities = []
        for x in predictions:
            probabilities.append([(1 - x), x])
        return probabilities

    def weights(self):
        return [0]
import sys
print("../../base")


from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer

class LogisticRegressionModel:
    def __init__(self, config):
        self.log = logging.getLogger("LogisticRegressionModel")
        self.name = "logistic_regression"
        self.model = LogisticRegressionCV(
                penalty='l2', scoring='roc_auc', tol=1e-4, n_jobs=-1, refit=True, random_state=11,
                verbose=1)
        self.log.info("inited")

    def load_train(self, x_train, y_train):
        self.log.info("load x_train size: {0} y_train size: {1}".format(x_train.shape[0], len(y_train)))
        self.model.fit(x_train, y_train)
        self.log.info("loaded")


    def predict_probabilities(self, x_to_predict):
        self.log.info("predict x_to_predict size: {0}".format(x_to_predict.shape[0]))
        return self.model.predict_proba(x_to_predict)

    def weights(self):
        return self.model.coef_[0]
import sys
print("../../base")








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
    if (name == "linear_svc"):
        return LinearSVCModel(model_config)
    logging.fatal("unknown model name: {0}".format(name))
import sys
print("../../base")


from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class SklearnCountVectorizerModel:
    def __init__(self, config):
        self.log = logging.getLogger("SkLearnCountVectorizerModel")
        self.name = "sklearn_count_vectorizer"
        self.max_features_count = config["max_features_count"]
        self.ngram_range = (config["ngram_range"][0], config["ngram_range"][1])
        self.count_vectorizer = text.CountVectorizer(
                ngram_range = self.ngram_range,
                max_features = self.max_features_count
        )
        self.log.info("inited")

    def load_train(self, x_train, y_train):
        print("size:", len(x_train), "tests:", x_train)
        print(y_train)
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(x_train)
        self.tfidf = self.vectorizer.transform(x_train)

        self.lr = LogisticRegression()
        self.lr.fit(self.tfidf, y_train)


    def predict(self, x_to_predict):
        return self.lr.predict_proba(self.vectorizer.transform(x_to_predict))


import sys
print("../../base")


class WordWeightModel:
    def __init__(self, config):
        self.log = logging.getLogger("WordWeightModel")
        self.name = "word_weight"
        self.min_weight_abs = config["min_weight_abs"]
        self.min_word_total_count = config["min_word_total_count"]
        self.unknown_state_radius = config["unknown_state_radius"]

    def load_train(self, x_train, y_train):
        negative_counts = {}
        positive_counts = {}
        for i in range(len(y_train)):
            for word, count in x_train[i].items():
                if (y_train[i] == 1):
                    if word not in positive_counts:
                        positive_counts[word] = 0
                    positive_counts[word] += count
                else:
                    if word not in negative_counts:
                        negative_counts[word] = 0
                    negative_counts[word] += count
            if (i % 10000 == 0):
                self.log.debug("loaded line: {0}/{1}".format(i, len(y_train)))
        self.word_weights = {}
        for word, count in negative_counts.items():
            total_count = count + positive_counts.get(word, 0)
            if (total_count >= self.min_word_total_count):
                self.word_weights[word] = (1 - count / total_count) * 2 - 1
        for word, count in positive_counts.items():
            total_count = count + negative_counts.get(word, 0)
            if (total_count >= self.min_word_total_count):
                self.word_weights[word] = (count / total_count) * 2 - 1

    def predict(self, x_to_predict):
        prediction = []
        positive_count = 0
        negative_count = 0
        unknown_count = 0
        for i in range(len(x_to_predict)):
            total_weight = 0
            trigger_words_count = 0
            for word, count in x_to_predict[i].items():
                if (word in self.word_weights and
                        abs(self.word_weights[word]) > self.min_weight_abs):
                    total_weight += self.word_weights[word] * count
                    trigger_words_count += 1
            if (trigger_words_count == 0):
                prediction.append(-1)
                unknown_count += 1
            else:
                total_weight /= trigger_words_count
                if total_weight > self.unknown_state_radius:
                    prediction.append(1)
                    positive_count += 1
                elif total_weight < -self.unknown_state_radius:
                    prediction.append(0)
                    negative_count += 1
                else:
                    prediction.append(-1)
                    unknown_count += 1
            if (i % 10000 == 0):
                self.log.debug("predicted line: {0}/{1}".format(i, len(x_to_predict)))
        self.log.info("predicted results: {0} positive {1} negative {2} unknown".format(
            positive_count, negative_count, unknown_count))
        return prediction


import sys
print("../base")
print("../data_provider")
print("../x_transformer")
print("../model")






def print_weights(x_transformer, model):
    f_weights = zip(x_transformer.features(), model.weights())
    f_weights = sorted(f_weights, key=lambda i: i[1])
    for i in range(1,10):
        print('%s, %.2f' % f_weights[-i])

    print('...')
    for i in reversed(range(1,10)):
        print('%s, %.2f' % f_weights[i])

def calculate_header_predictions(config):
    data_provider = DataProvider(config["header_data_provider"], "header")
    x_transformer = x_transformer_by_config(config, True)
    model = model_by_config(config)

    x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
    model.load_train(x_transformer.transform(data_provider.x_train), data_provider.y_train)

    prediction = model.predict_probabilities(x_transformer.transform(data_provider.x_to_predict))

    """
    print("header:")
    print_weights(x_transformer, model)
    """

    return prediction

def calculate_body_predictions(config):
    data_provider = DataProvider(config["body_data_provider"], "body")
    x_transformer = x_transformer_by_config(config, False)
    model = model_by_config(config)

    x_transformer.load_train_data(data_provider.x_train, data_provider.y_train)
    model.load_train(x_transformer.transform(data_provider.x_train), data_provider.y_train)

    prediction = model.predict_probabilities(x_transformer.transform(data_provider.x_to_predict))

    """
    print("body:")
    print_weights(x_transformer, model)
    """

    return prediction

def calculate_final_prediction(header_prediction, body_prediction, config, log):
    final_prediction = []
    header_weight = config["header_weight"]
    negative_count = positive_count = unknown_count = 0
    min_decision_probability = config["min_decision_probability"]
    for i in range(len(header_prediction)):
        negative_probability = sqrt(header_prediction[i][0] * body_prediction[i][0])
        positive_probability = sqrt(header_prediction[i][1] * body_prediction[i][1])
        """
        negative_probability = (header_prediction[i][0] * header_weight + body_prediction[i][0] * (1.0 - header_weight)) / 2.0
        positive_probability = (header_prediction[i][1] * header_weight + body_prediction[i][1] * (1.0 - header_weight)) / 2.0
        """
        if (positive_probability >= min_decision_probability):
            final_prediction.append(1)
            positive_count += 1
            continue
        if (negative_probability >= min_decision_probability):
            final_prediction.append(0)
            negative_count += 1
            continue

        final_prediction.append(positive_probability)
        unknown_count += 1

    log.info("predicted results: positive: {0} negative: {1} unknown: {2}".format(
        positive_count, negative_count, unknown_count))
    return final_prediction

def flush_wrong_predictions(expected, final_prediction, config, header_prediction, body_prediction):
    x_to_predict = open(config["x_to_predict"]).readlines()
    wrong_predictions_file = open(config["wrong_predictions_file"], "w")
    header_weight = config["header_weight"]
    for i in range(len(final_prediction)):
        if final_prediction[i] != -1 and final_prediction[i] != expected[i]:
            negative_probability = header_prediction[i][0] * body_prediction[i][0]
            positive_probability = header_prediction[i][1] * body_prediction[i][1] 
            wrong_predictions_file.write(
                "line: %s expected: %s predicted: %s len: %s\nheader_prediction: %s body_prediction: %s negative_probability: %s positive_probability: %s\n    %s\n" % 
                    (i + 2, expected[i], final_prediction[i], len(x_to_predict[i]),
                        header_prediction[i], body_prediction[i], negative_probability,
                        positive_probability, x_to_predict[i]))

log = logging.getLogger("Launcher")

config_file = open("../input/config.json", 'r')
config = json.load(config_file)
log.info("launcher config: {0}".format(config))

header_prediction = calculate_header_predictions(config)
body_prediction = calculate_body_predictions(config)
final_prediction = calculate_final_prediction(header_prediction, body_prediction, config, log)

answer_file = open(config["answer_file"], 'w')
answer_file.write("Id,Probability\n")
for i in range(len(final_prediction)):
    answer_file.write("%s,%s\n" % (i + 1, final_prediction[i]))

if config["need_magic"]:
    expected = pd.read_csv(config["magic"])["Probability"].tolist()

    print("ratio_score:", ratio_score(expected, final_prediction))
    flush_wrong_predictions(expected, final_prediction, config, header_prediction, body_prediction)

    answer_file = open(config["answer_file"], "w")

