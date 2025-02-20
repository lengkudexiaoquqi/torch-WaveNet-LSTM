"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.
"""

import os
from pytorchtools import EarlyStopping
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
from WaveNet_LSTM_AQP.WaveNet_LSTM import *
import numpy as np
from optuna.trial import TrialState
from data_preparation import Data_Split
import warnings


warnings.filterwarnings('ignore')
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
set_seed(1)
DEVICE = torch.device("cpu")
BATCH_SIZE = 27
DIR = os.getcwd()


set_seed(1)
EPOSCHS = 1158
BATCH_SIZE = 27

X_pos = [0,11]
Y_pos = 0
Window_size = 8
FEATURE = len(X_pos)
RES_FILTER= 122
SKIP_FILTER=195
DILATED= 3
LSTM_UNIT= 156
LR= 2.1121953850951244e-05
# patience =10
# early_stopping =EarlyStopping(patience,verbose=True)



dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=Window_size, predict_step=1, filter_size=50,
                     feature=FEATURE)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

lstm = WaveNet_LSTM(input_size=FEATURE, residual_size=RES_FILTER, skip_size=SKIP_FILTER,
                    dilation_depth=DILATED, hidden_size=LSTM_UNIT)
# lstm.to(device)
weight_p, bias_p = [],[]
for name, p in lstm.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
# optimizer_lstm = torch.optim.Adam([{'params': weight_p, 'weight_decay':0},
#                       {'params': bias_p, 'weight_decay':0}],
#                       lr=LR)
optimizer_lstm = torch.optim.Adam(lstm.parameters(),lr=LR)

loss_func = nn.MSELoss()
# training and testing
for epoch in range(EPOSCHS):
    lstm.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = torch.tensor(b_x, dtype=torch.float32)
        b_y = torch.tensor(b_y, dtype=torch.float32)

        outout = lstm(b_x)
        loss = loss_func(outout, b_y)
        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()


# lstm.to(device='cpu')
dataset.X_test_scaler = torch.tensor(dataset.X_test_scaler, dtype=torch.float32)
test_output = lstm(dataset.X_test_scaler)
y_pred = test_output.detach().numpy()
# 反归一化
y_pred = dataset.scaler.inverse_transform(y_pred)
y_true = dataset.y_test
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mape = MAPE(y_true, y_pred)
rmse = RMSE(y_true=y_true, y_pred=y_pred)
r2 = r2_score(y_true=y_true, y_pred=y_pred)
# torch.save(lstm,"./AQI.pkl")
# np.save("./AQI.npy",y_pred)
print("current_param: SKIP_FILTER :{0} RES_FILTER :{1} DILATED :{2}  LSTM_UNITS:{3} Learning_rate: {4}".format(
    SKIP_FILTER, RES_FILTER, DILATED, LSTM_UNIT, LR))
print("mae={0:.4f},mse={1:.4f},r2={2:.4f},rmse={3:.4f},mape={4:.4f}".format(mae, mse, r2, rmse, mape))
print("-------------------------------------------------")