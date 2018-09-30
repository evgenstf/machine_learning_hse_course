import sklearn.linear_model as lm

def build_model(x_train, y_train):
    model = lm.Ridge()
    model.fit(x_train, y_train)
    return model
