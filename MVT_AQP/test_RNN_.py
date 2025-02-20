"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.
"""

import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torch.utils.data
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from seed_set import set_seed
import optuna
from MVT_AQP.BaseLine_MVT import *
import numpy as np
from optuna.trial import TrialState
from data_preparation import Data_Split
import warnings


warnings.filterwarnings('ignore')
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
DEVICE = torch.device("cpu")
BATCH_SIZE = 150
DIR = os.getcwd()
set_seed(1)

def objective(HIDDEN_SIZE,LR,filename):
    set_seed(1)
    global X_pos
    global Y_pos
    global Window_size
    global FILTER_SIZE

    EPOSCHS = 1000

    FEATURE=len(X_pos)
    dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=Window_size, predict_step=1, filter_size=FILTER_SIZE, feature=FEATURE)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)


    rnn = RNN(hidden_size=HIDDEN_SIZE,predict_step=1,input_dim=FEATURE)
    optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    # training and testing
    for epoch in range(EPOSCHS):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = torch.tensor(b_x, dtype=torch.float32)
            b_y = torch.tensor(b_y, dtype=torch.float32)

            outout = rnn(b_x)
            loss = loss_func(outout, b_y)
            optimizer_rnn.zero_grad()
            loss.backward()
            optimizer_rnn.step()

    dataset.X_test_scaler = torch.tensor(dataset.X_test_scaler, dtype=torch.float32)
    test_output = rnn(dataset.X_test_scaler)
    y_pred = test_output.detach().numpy()
    # 反归一化
    y_pred = dataset.scaler.inverse_transform(y_pred)
    y_true = dataset.y_test
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    rmse = RMSE(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    torch.save(rnn,"../Diebold-Mariano-Test-master/muti_factor_predict_value/rnn/{}.pkl".format(filename))
    np.save("../Diebold-Mariano-Test-master/muti_factor_predict_value/rnn/{}.npy".format(filename),y_pred)
    print("current_hidden_num----> {0}   current_lr------>{1}".format(HIDDEN_SIZE,LR))
    print("mae={0:.4f},mse={1:.4f},r2={2:.4f},rmse={3:.4f},mape={4:.4f}".format(mae, mse, r2, rmse, mape))
    print("-------------------------------------------------")
    return mape


if __name__ == "__main__":
    # 让结果可复现
    set_seed(1)
    RNN_hyper={
        'AQI': {
            "HIDDEN_SIZE":63,
            "LR":0.001033486949035015
        },
        'PM10':{
            "HIDDEN_SIZE": 15,
            "LR": 0.002240309277159458
        },
        'CO':{
            "HIDDEN_SIZE": 21,
            "LR": 0.0028942745657248616
        },
        "O3":{
            "HIDDEN_SIZE": 33,
            "LR":0.0019980795521009733
        },
        "NO2":{
            "HIDDEN_SIZE":14,
            "LR":0.001088133500363589
        },
        "SO2":{
            "HIDDEN_SIZE": 63,
            "LR": 0.00247908992385815
        },
        "PM2.5":{
            "HIDDEN_SIZE": 36,
            "LR":0.002989100454861537
        }
    }

    X_pos = [0,11]
    Y_pos = 0
    FILTER_SIZE =50
    Window_size = 8
    HIDDEN_SIZE = 63
    LR = 0.001033486949035015
    objective(HIDDEN_SIZE,LR,"AQI")

