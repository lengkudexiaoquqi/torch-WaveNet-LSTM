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
import torch.utils.data
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from seed_set import set_seed
import optuna
from BaseLine_AQP.BaseLine import *
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


HIDDEN_SIZE = 44
EPOSCHS = 1000
LR = 0.0020780045115612986



X_POS=[0,11]
Y_POS=0
Window_size=8
filter_size=50
FEATURE = len(X_POS)*Window_size

dataset = Data_Split(x_pos=X_POS, y_pos=Y_POS, window_size=Window_size, predict_step=1, filter_size=filter_size)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

bp = LSTM(hidden_size=HIDDEN_SIZE, predict_step=1)
optimizer_bp = torch.optim.Adam(bp.parameters(),lr=LR)
loss_func = nn.MSELoss()
# training and testing
for epoch in range(EPOSCHS):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = torch.tensor(b_x, dtype=torch.float32)
        b_y = torch.tensor(b_y, dtype=torch.float32)

        outout = bp(b_x)
        loss = loss_func(outout, b_y)
        optimizer_bp.zero_grad()
        loss.backward()
        optimizer_bp.step()


dataset.X_test_scaler = torch.tensor(dataset.X_test_scaler, dtype=torch.float32)
test_output = bp(dataset.X_test_scaler)
y_pred = test_output.detach().numpy()
# 反归一化
y_pred = dataset.scaler.inverse_transform(y_pred)
y_true = dataset.y_test


mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mape = MAPE(y_true, y_pred)
rmse = RMSE(y_true=y_true, y_pred=y_pred)
r2 = r2_score(y_true=y_true, y_pred=y_pred)
mape = MAPE(y_true, y_pred)
print("mae={0:.4f},mse={1:.4f},r2={2:.4f},rmse={3:.4f},mape={4:.4f}".format(mae, mse, r2, rmse, mape))