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

        #np.random.seed(seed=0)
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


        for i in config["features_to_multiply"]:
            for j in config["features_to_multiply"]:
                if i == j:
                    continue
                self.x_known = np.concatenate((self.x_known, (self.x_known[:, i] * self.x_known[:, j]).reshape(-1, 1)), axis=1)
                self.x_known = np.concatenate((self.x_known, (np.log(self.x_known[:, i]) * np.log(self.x_known[:, j])).reshape(-1, 1)), axis=1)

                self.x_to_predict = np.concatenate((self.x_to_predict, (self.x_to_predict[:, i] * self.x_to_predict[:, j]).reshape(-1, 1)), axis=1)
                self.x_to_predict = np.concatenate((self.x_to_predict, (np.log(self.x_to_predict[:, i]) * np.log(self.x_to_predict[:, j])).reshape(-1, 1)), axis=1)

        self.log.info("loaded x_known rows: {0} columns: {1}".format(self.x_known.shape[0], self.x_known.shape[1]))
        self.log.info("loaded x_to_predict rows: {0} columns: {1}".format(self.x_to_predict.shape[0], self.x_to_predict.shape[1]))


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

