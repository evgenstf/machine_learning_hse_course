import sys
sys.path.append("../base")
sys.path.append("../data_provider")
sys.path.append("../models")
from common import *
from data_provider import DataProvider
import simple_linear_regression as simple_lr
import lasso_linear_regression as lasso_lr
import ridge_linear_regression as ridge_lr
import catboost_regression as catboost_r

data_provider = DataProvider()
x_raw = data_provider.x_raw
y_raw = data_provider.y_raw

x_train = data_provider.processed_x_train()
y_train = data_provider.processed_y()

def train_and_test_simple_model():
    simple_model = simple_lr.build_model(x_train, y_train)
    train_prediction = data_provider.revert_y(simple_model.predict(x_train))
    print("simple model train prediction mape score:", mape_score(y_raw, train_prediction))

def train_and_test_lasso_model():
    lasso_model = lasso_lr.build_model(x_train, y_train)

    train_prediction = data_provider.revert_y(lasso_model.predict(x_train))
    print("lasso model train prediction mape score:", mape_score(y_raw, train_prediction))

def train_and_test_ridge_model():
    ridge_model = ridge_lr.build_model(x_train, y_train)

    train_prediction = data_provider.revert_y(ridge_model.predict(x_train))
    print("ridge model train prediction mape score:", mape_score(y_raw, train_prediction))

def train_and_test_catboost_model():
    catboost_model = catboost_r.build_model(x_train, y_train)

    train_prediction = data_provider.revert_y(catboost_model.predict(x_train))
    print("catboost model train prediction mape score:", mape_score(y_raw, train_prediction))

    test_prediction = catboost_model.predict(data_provider.processed_x_test())
    test_prediction = data_provider.revert_y(test_prediction)
    for item in test_prediction:
        if item < 0:
            print("fail!")
    answer_file = open("result.csv", "w")

    answer_file.write("Id,Price\n")
    for i in range(len(test_prediction)):
        answer_file.write("%s,%s\n" % (i + 1, test_prediction[i]))

train_and_test_catboost_model()
