import sklearn.linear_model as lm

def build_model():
    model = lm.Ridge()
    return model

def build_model_with_custom_alpha(alpha):
    model = lm.Ridge(alpha)
    return model
