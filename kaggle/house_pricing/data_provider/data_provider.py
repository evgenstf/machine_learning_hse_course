import pandas as pd
import numpy as np
import math
import sys
import sklearn.preprocessing as pr
from sklearn.model_selection import train_test_split

class DataProvider(object):
    def __init__(self):
#       self.x_raw = pd.DataFrame(np.load('../data/raw/x_test.npy'))
#       self.y_raw = pd.read_csv('../data/raw/correct_kek.csv').drop(columns="Id")
#       self.x_control_test = pd.DataFrame(np.load('../data/raw/x_test.npy'))

        self.x_raw = pd.DataFrame(np.load('data/x_train.npy'))
        self.y_raw = pd.DataFrame(np.load('data/y_train.npy'))
        self.x_control_test = pd.DataFrame(np.load('data/x_test.npy'))

        print("all data loaded")

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_raw, self.y_raw, train_size=0.9, random_state=42
        )
#        self.x_train, self.x_test, self.y_train, self.y_test = x_raw, x_raw, y_raw, y_raw
        self.x_train = self.process_x(self.x_train)
        self.x_test = self.process_x(self.x_test)
        self.x_control_test = self.process_x(self.x_control_test)
        self.y_train = self.process_y(self.y_train)
        self.y_test = self.process_y(self.y_test)

        print("all data processed")

#       self.x_raw = pd.DataFrame(np.load('../data/raw/x_test.npy'))
#       self.y_raw = pd.read_csv('../data/raw/correct_kek.csv').drop(columns="Id")
#       self.x_control_test = pd.DataFrame(np.load('../data/raw/x_test.npy'))

    def process_x(self, x_raw):
        result = x_raw.copy()
        result['date'] = pd.to_numeric(pd.to_datetime(x_raw['date'], format='%Y-%m-%d', utc=True))
        result['date'] /= 100000000000000

        #result['grade'] = np.power(x_raw['grade'], 3)
        result['sqft_living'] = np.log2(x_raw['sqft_living'])
        result['sqft_above'] = np.log2(x_raw['sqft_above'])
        #result['bedrooms'] = np.log2(x_raw['bedrooms'])

        #result['lat'] = x_raw['lat'].apply(
        #        lambda x : min(x, 2 * lat_mirror_value - x) ** 5
        #)

        result['lat1'] = result['lat'].apply(
                lambda x: x if x < 47.52 else np.nan
        )
        result['lat2'] = result['lat'].apply(
                lambda x: x if 47.52 <= x and x <= 47.62 else np.nan
        )
        result['lat3'] = result['lat'].apply(
                lambda x: x if 47.62 < x else np.nan
        )

        result['bedrooms'] = x_raw['bedrooms'].apply(
                lambda x : np.log2(min(2 ** min(x, 13), 100))
        )
        

        conditions = []
        for condition in x_raw["condition"]:
            conditions.append(condition)
        conditions = np.unique(conditions)
        for condition in conditions:
            result[str(condition) + 'condition'] = x_raw['condition'] == condition

        bedrooms = []
        for bedroom in x_raw["bedrooms"]:
            bedrooms.append(bedroom)
        bedrooms = np.unique(bedrooms)
        for bedroom in bedrooms:
            result[str(bedroom) + 'bedrooms'] = x_raw['bedrooms'] == bedroom

        zipcodes = []
        for zipcode in x_raw["zipcode"]:
            zipcodes.append(zipcode)
        zipcodes = np.unique(zipcodes)
        for zipcode in zipcodes:
            result[str(zipcode) + 'code'] = x_raw['zipcode'] == zipcode

        floors = []
        for floor in x_raw["floors"]:
            floors.append(floor)
        floors = np.unique(floors)
        for floor in floors:
            result[str(floor) + 'floor'] = x_raw['floors'] == floor

        result['yr_renovated'] = result['yr_renovated'].replace(0, np.nan)
        result['sqft_basement'] = result['sqft_basement'].replace(0, np.nan)
        result['sqft_basement'] = result['sqft_basement'].apply(
                lambda x : math.sqrt(x)
        )

        result['yr_built1'] = result['yr_built'].apply(
                lambda x: x if x < 1945 else np.nan
        )

        result['yr_built2'] = result['yr_built'].apply(
                lambda x: x if x >= 1945 else np.nan
        )

        result['sqft_lot'] = result['sqft_lot'].apply(
                lambda x: math.log2(x)
        )
        result['sqft_lot1'] = result['sqft_lot'].apply(
                lambda x: x if 13 <= x and x <= 14 else np.nan
        )
#        result = result.drop(columns=['lat', 'yr_built', 'sqft_lot', 'zipcode', 'long', 'date',
#                'floors', 'yr_renovated'])
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

    def train_data_by_columns(self, columns):
        x_result = self.x_train.copy()
        x_result['target'] = self.y_train
        for column in columns:
            if column in x_result.columns:
                x_result = x_result.dropna(subset = [ column ])
        x_result = x_result.dropna(axis='columns')
        y_result = x_result['target']
        x_result = x_result.drop(columns=['target'])
        return (x_result, y_result)

    def process_prediction(self, prediction):
        prediction = self.revert_y(prediction)
        popular_prices = self.y_raw.as_matrix()
        result = []
        for x in prediction:
            result.append(min(popular_prices, key = lambda y : abs(y - x)))

        return prediction
