import pandas as pd
import numpy as np
import math
import sys
import sklearn.preprocessing as pr
from sklearn.model_selection import train_test_split

class DataProvider(object):
    def __init__(self):
        self.x_raw = pd.DataFrame(np.load('../data/raw/x_train.npy'))
        self.y_raw = pd.DataFrame(np.load('../data/raw/y_train.npy'))
        self.x_control_test = pd.DataFrame(np.load('../data/raw/x_test.npy'))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_raw, self.y_raw, train_size=0.5, random_state=42
        )

        self.x_train = self.process_x(self.x_train)
        self.x_test = self.process_x(self.x_test)
        self.x_control_test = self.process_x(self.x_control_test)
        self.y_train = self.process_y(self.y_train)
        self.y_test = self.process_y(self.y_test)

    def process_x(self, x_raw):
        result = x_raw.copy()
        result['date'] = pd.to_numeric(pd.to_datetime(x_raw['date'], format='%Y-%m-%d', utc=True))
        result['date'] /= 100000000000000

        #result['grade'] = np.power(x_raw['grade'], 3)
        result['sqft_living'] = np.log2(x_raw['sqft_living'])
        result['sqft_above'] = np.log2(x_raw['sqft_above'])
        #result['bedrooms'] = np.log2(x_raw['bedrooms'])
        lat_mirror_value = 47.65
        result['lat'] = x_raw['lat'].apply(
                lambda x : min(x, 2 * lat_mirror_value - x) ** 5
        )
        result['bedrooms'] = x_raw['bedrooms'].apply(
                lambda x : np.log2(min(2 ** min(x, 13), 100))
        )
        result['1floor'] = x_raw['floors'] == 1
        result['2floor'] = x_raw['floors'] == 2

        zipcodes = []
        for zipcode in x_raw["zipcode"]:
            zipcodes.append(zipcode)
        zipcodes = np.unique(zipcodes)
        for zipcode in zipcodes:
            result[str(zipcode) + 'code'] = x_raw['zipcode'] == zipcode

        #result = result.drop(columns=['date', 'long', 'sqft_lot', 'floors', 'zipcode'])
        return result

    def process_y(self, y_raw):
        result = y_raw.copy()
        result = np.log2(y_raw)
        return result

    def revert_y(self, y_processed):
        result = []
        for x in y_processed:
            result.append(2 ** x)
        return result
