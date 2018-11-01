import sys
sys.path.append("../../base")
from common import *

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


