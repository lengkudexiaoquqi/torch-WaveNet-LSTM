import numpy as np
from sklearn.metrics import *

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def SMAPE2(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    smape = SMAPE2(y_true, y_pred)
    rmse = RMSE(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    return mae, mse, mape, smape, rmse, r2