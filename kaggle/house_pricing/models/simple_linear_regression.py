import sklearn.linear_model as lm

def build_model(x_train, y_train):
    model = lm.LinearRegression()
    model.fit(x_train, y_train)
    return model
