import sys
sys.path.append("../base")
from common import *

class DataReader:
    def __init__(self, config):
        self.log = logging.getLogger("DataReader")
        self.log.info("data reader config: {0}".format(config))
        self.data_path = config["data_path"]
        with open(self.data_path) as file:
            self.lines = file.readlines()
        self.log.info("loaded {0} reviews".format(len(self.lines)))
        self.next_review_index = 0

    def has_review(self):
        return self.next_review_index < len(self.lines)

    def next_review(self):
        review = self.lines[self.next_review_index]
        self.next_review_index += 1
        return review
