from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, \
    Normalizer, FunctionTransformer, RobustScaler, QuantileTransformer
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf

class Predictor(object):
    def __init__(self, model, data_provider):
        self.model = model
        self.data_provider = data_provider

    def predict(self, x_control_test, prefix_size):
        predictions = []
        print("start prediction")
        for i in range(prefix_size):
            test_data = x_control_test[i : i + 1]
            test_data = test_data.dropna(axis = 'columns')
            columns = list(test_data.columns.values)
            x_train, y_train = self.data_provider.train_data_by_columns(columns)

            columns_to_drop = []
            for column in test_data.columns.values:
                if not column in x_train.columns.values:
                    columns_to_drop.append(column)
            test_data = test_data.drop(columns = columns_to_drop)

            columns_to_drop = []
            for column in x_train.columns.values:
                if not column in test_data.columns.values:
                    columns_to_drop.append(column)
            x_train = x_train.drop(columns = columns_to_drop)

            param_grid = {}

            pipeline = Pipeline([
                ('scaler', MinMaxScaler()),
                ('cl', RidgeCV())
            ])

            grid_searcher = GridSearchCV(pipeline, param_grid=param_grid)
            grid_searcher.fit(x_train, y_train)

            prediction = self.data_provider.process_prediction(grid_searcher.predict(test_data))
            predictions.append(prediction)

            if i % 10 == 0:
                print("processed", i, "/", len(x_control_test))
        return predictions
