import sys
sys.path.append("../base")
sys.path.append("../data_provider")
sys.path.append("../models")
from common import *
from data_provider import DataProvider
from predictor import Predictor
import simple_linear_regression as simple_lr
import lasso_linear_regression as lasso_lr
import ridge_linear_regression as ridge_lr

data_provider = DataProvider()
x_raw = data_provider.x_raw
y_raw = data_provider.y_raw

x_train = data_provider.x_train
y_train = data_provider.y_train

x_control_test = data_provider.x_control_test
#y_control_test = data_provider.y_control_test

def train_and_test_simple_model():
    simple_model = simple_lr.build_model(x_train, y_train)
    train_prediction = data_provider.revert_y(simple_model.predict(x_train))
    print("simple model train prediction mape score:", mape_score(y_raw, train_prediction))

def train_and_test_lasso_model():
    lasso_model = lasso_lr.build_model(x_train, y_train)

    train_prediction = data_provider.revert_y(lasso_model.predict(x_train))
    print("lasso model train prediction mape score:", mape_score(y_raw, train_prediction))

def train_and_test_ridge_model():
    predictor = Predictor(Ridge(), data_provider)

    train_prediction = predictor.predict(x_control_test, len(x_control_test))
    answer_file = open("solution-1-EvgeniiKazakov.csv", "w")
    answer_file.write("Id,Price\n")
    for i in range(len(train_prediction)):
        prediction = train_prediction[i]
        answer_file.write("%s,%s\n" % (i + 1, prediction[0]))
    #print("ridge model train prediction mape score:", mape_score(y_control_test[:len(train_prediction)], train_prediction))

train_and_test_ridge_model()
