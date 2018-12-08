import sys
sys.path.append("../base")
from common import *
from sklearn.model_selection import train_test_split
from random import shuffle
import pandas as pd




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

        self.log.debug("loaded {0} x_known lines".format(len(self.x_known)))
        self.log.debug("loaded {0} y_known lines".format(len(self.y_known)))

        x_to_predict_archive = np.load(self.x_to_predict_path)
        self.x_to_predict = x_to_predict_archive[x_to_predict_archive.files[0]]
        self.log.debug("loaded {0} x_to_predict lines".format(len(self.x_to_predict)))

        self.log.info("inited")

