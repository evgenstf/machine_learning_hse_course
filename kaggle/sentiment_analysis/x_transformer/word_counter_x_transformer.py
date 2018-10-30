import sys
sys.path.append("../../base")
from common import *
import string

class WordCounterXTransformer:
    def __init__(self, config):
        self.log = logging.getLogger("WordCounterXTransformer")
        self.log.info("x_transformer config:", config)

        self.name = "word_counter"
        self.min_word_len = config["min_word_len"]
        self.min_word_total_count = config["min_word_total_count"]
        self.log.info("inited")

    def load_train(self, x_train, y_train):
        negative_counts = {}
        positive_counts = {}
        for i in range(len(y_train)):
            for word in [word.strip(string.punctuation) for word in x_train[i].split()]:
                if (len(word) >= self.min_word_len):
                    if (y_train[i] == 1):
                        if word not in positive_counts:
                            positive_counts[word] = 0
                        positive_counts[word] += 1
                    else:
                        if word not in negative_counts:
                            negative_counts[word] = 0
                        negative_counts[word] += 1
            if (i % 10000 == 0):
                self.log.debug("processed line: {0}".format(i))
        self.word_weights = {}
        for word, count in negative_counts.items():
            total_count = count + positive_counts.get(word, 0)
            if (total_count >= self.min_word_total_count):
                self.word_weights[word] = (1 - count / total_count) * 2 - 1
        for word, count in positive_counts.items():
            total_count = count + negative_counts.get(word, 0)
            if (total_count >= self.min_word_total_count):
                self.word_weights[word] = (count / total_count) * 2 - 1
