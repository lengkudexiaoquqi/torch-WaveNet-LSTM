from data_preparation import Data_Split
from BaseLine_AQP.BaseLine import *
import pandas as pd
from evaluation_metrics import *
import warnings
from seed_set import set_seed
import global_constants as gc

warnings.filterwarnings('ignore')

indicators = ["AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AT", "ATM", "WDIR", "WFS", "PRCP", "RH"]

predict_step = gc.STEP()
poses = gc.POSES()
EPOSCHS = 500
BATCH_SIZE = 150

results = pd.DataFrame(columns=['Method', 'Indicator', 'MSE', 'MAPE', 'RMSE', 'R2', 'MAE', 'SMAPE'])

for p in poses:
    X_pos = poses[p]
    Y_pos = X_pos

    set_seed(1)

    WINDOW_SIZE = gc.OPTIMAL_WINDOW_SIZE(X_pos)
    HIDDEN_SIZE = int((WINDOW_SIZE + predict_step) * 2 / 3)
    dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=WINDOW_SIZE, predict_step=1,
                            filter_size=gc.SSA_FILTER_COMPONENT(X_pos))
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)  # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。

    bp = BPNet(window_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE, predict_step=1)
    rnn = RNN(hidden_size=HIDDEN_SIZE, predict_step=1)
    lstm = LSTM(hidden_size=HIDDEN_SIZE, predict_step=1)

    optimizer_bp = torch.optim.Adam(bp.parameters(), lr=0.01)
    optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=0.01)
    optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=0.01)
    loss_bp_func = nn.MSELoss()
    loss_rnn_func = nn.MSELoss()
    loss_lstm_func = nn.MSELoss()
    # training and testing
    for epoch in range(EPOSCHS):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = torch.as_tensor(b_x, dtype=torch.float32)
            b_y = torch.as_tensor(b_y, dtype=torch.float32)

            out_bp = bp(b_x)
            loss_bp = loss_bp_func(out_bp, b_y)
            optimizer_bp.zero_grad()
            loss_bp.backward()
            optimizer_lstm.step()

            out_rnn = rnn(b_x)
            loss_rnn = loss_rnn_func(out_rnn, b_y)
            optimizer_rnn.zero_grad()
            loss_rnn.backward()
            optimizer_rnn.step()

            out_lstm = lstm(b_x)
            loss_lstm = loss_rnn_func(out_lstm, b_y)
            optimizer_lstm.zero_grad()
            loss_lstm.backward()
            optimizer_bp.step()

    dataset.X_test_scaler = torch.tensor(dataset.X_test_scaler, dtype=torch.float32)
    y_true = dataset.y_test

    test_output_bp = bp(dataset.X_test_scaler)
    bp_y_pred = test_output_bp.detach().numpy()
    # 反归一化
    bp_y_pred = dataset.scaler.inverse_transform(bp_y_pred)

    test_output_rnn = rnn(dataset.X_test_scaler)
    rnn_y_pred = test_output_rnn.detach().numpy()
    # 反归一化
    rnn_y_pred = dataset.scaler.inverse_transform(rnn_y_pred)

    test_output_lstm = lstm(dataset.X_test_scaler)
    lstm_y_pred = test_output_lstm.detach().numpy()
    # 反归一化
    lstm_y_pred = dataset.scaler.inverse_transform(lstm_y_pred)

    bp_mae, bp_mse, bp_mape, bp_smape, bp_rmse, bp_r2 = evaluate(y_true, bp_y_pred)
    results = results.append([{'Method': 'BP', 'Indicator': indicators[p], 'MSE': bp_mse, 'MAPE': bp_mape, 'RMSE': bp_rmse, 'R2': bp_r2, 'MAE': bp_mae, 'SMAPE': bp_smape}])

    rnn_mae, rnn_mse, rnn_mape, rnn_smape, rnn_rmse, rnn_r2 = evaluate(y_true, rnn_y_pred)
    results = results.append([{'Method': 'RNN', 'Indicator': indicators[p], 'MSE': rnn_mse, 'MAPE': rnn_mape, 'RMSE': rnn_rmse, 'R2': rnn_r2, 'MAE': rnn_mae, 'SMAPE': rnn_smape}])

    lstm_mae, lstm_mse, lstm_mape, lstm_smape, lstm_rmse, lstm_r2 = evaluate(y_true, lstm_y_pred)
    results = results.append([{'Method': 'LSTM', 'Indicator': indicators[p], 'MSE': lstm_mse, 'MAPE': lstm_mape, 'RMSE': lstm_rmse, 'R2': lstm_r2, 'MAE': lstm_mae, 'SMAPE': lstm_smape}])

    print(indicators[p], "is done!")

results.to_csv("singleBasline_results.csv".format(indicators[X_pos]))
