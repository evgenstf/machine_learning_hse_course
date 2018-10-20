import sklearn.linear_model as lm

def build_model():
    model = lm.Lasso(0.01)
    return model

def build_model_with_custom_alpha(alpha):
    model = lm.Lasso(alpha)
    return model
