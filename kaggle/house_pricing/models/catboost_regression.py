import numpy as np
from catboost import CatBoostRegressor

def build_model(x_train, y_train):
    model = CatBoostRegressor(learning_rate=0.1, iterations=100, random_seed=0)
    model.fit(x_train, y_train)
    return model
