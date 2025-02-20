"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.
"""

import os
import warnings
import torch.utils.data
import global_constants as gc
import pandas as pd

from evaluation_metrics import *
from WaveNet_LSTM_AQP.WaveNet_LSTM import *
from data_preparation import Data_Split
from seed_set import set_seed

warnings.filterwarnings('ignore')

set_seed(1)
DEVICE = torch.device("cpu")
BATCH_SIZE = 150
DIR = os.getcwd()

set_seed(1)
EPOSCHS = 100
BATCH_SIZE = 150

results = pd.DataFrame(columns=['Indicator', 'MSE', 'MAPE', 'RMSE', 'R2', 'MAE', 'SMAPE'])

for p in gc.POSES():
    Y_pos = p
    X_pos = gc.CAUSAL_FACTORS(Y_pos)
    Window_size = gc.OPTIMAL_WINDOW_SIZE(Y_pos)

    input_feature_dim = len(X_pos)
    RES_FILTER = 120
    SKIP_FILTER = 183
    dilated_length = 2
    lstm_hidden_unit = 32
    lr = 0.001
    # patience =10
    # early_stopping =EarlyStopping(patience,verbose=True)

    dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=Window_size, predict_step=1, filter_size=gc.SSA_FILTER_COMPONENT(p),
                         feature=input_feature_dim)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

    wavnetLstm = WaveNet_LSTM(input_size=input_feature_dim, residual_size=RES_FILTER, skip_size=SKIP_FILTER,
                              dilation_depth=dilated_length, hidden_size=lstm_hidden_unit)
    # lstm.to(device)
    weight_p, bias_p = [], []
    for name, p in wavnetLstm.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    # optimizer_lstm = torch.optim.Adam([{'params': weight_p, 'weight_decay':0},
    #                       {'params': bias_p, 'weight_decay':0}],
    #                       lr=LR)
    optimizer_lstm = torch.optim.Adam(wavnetLstm.parameters(), lr=lr)

    loss_func = nn.MSELoss()
    # training and testing
    for epoch in range(EPOSCHS):
        wavnetLstm.train()
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = torch.as_tensor(b_x, dtype=torch.float32)
            b_y = torch.as_tensor(b_y, dtype=torch.float32)

            outout = wavnetLstm(b_x)
            loss = loss_func(outout, b_y)
            optimizer_lstm.zero_grad()
            loss.backward()
            optimizer_lstm.step()

    # lstm.to(device='cpu')
    dataset.X_test_scaler = torch.as_tensor(dataset.X_test_scaler, dtype=torch.float32)
    test_output = wavnetLstm(dataset.X_test_scaler)
    y_pred = test_output.detach().numpy()
    # 反归一化
    y_pred = dataset.scaler.inverse_transform(y_pred)
    y_true = dataset.y_test
    mae, mse, mape, smape, rmse, r2 = evaluate(y_true, y_pred)
    print(gc.INDICATORS()[Y_pos],' is done!')
    results = results.append(
        [{'Indicator': gc.INDICATORS()[Y_pos], 'MAPE': mape, 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'SMAPE': smape, 'MSE': mse}])
    print(results)

results.to_csv("basicWaveNetLSTMResults.csv", encoding='utf-8')