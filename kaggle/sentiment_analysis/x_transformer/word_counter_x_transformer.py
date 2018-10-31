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
        self.log.info("inited")

    def transform(self, x_data):
        transformed_data = []
        for i in range(len(x_data)):
            counts = {}
            for word in [word.strip(string.punctuation) for word in x_data[i].split()]:
                if (len(word) >= self.min_word_len):
                        if word not in counts:
                            counts[word] = 0
                        counts[word] += 1
            transformed_data.append(counts)
            if (i % 10000 == 0):
                self.log.debug("transformed line: {0}/{1}".format(i, len(x_data)))
        return transformed_data
