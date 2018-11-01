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
