from pycaret.regression import *


def regr_predict(data, model='model'):
    model = load_model(model)
    return predict_model(model, data)
