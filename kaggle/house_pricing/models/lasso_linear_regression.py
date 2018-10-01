import sklearn.linear_model as lm

def build_model(x_train, y_train):
    model = lm.Lasso(0.01)
    model.fit(x_train, y_train)
    return model

def build_model_with_custom_alpha(x_train, y_train, alpha):
    model = lm.Lasso(alpha)
    model.fit(x_train, y_train)
    return model
