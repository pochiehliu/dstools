from pycaret.classification import *


def clas_predict(data, model='model'):
    model = load_model(model)
    return predict_model(model, data)
