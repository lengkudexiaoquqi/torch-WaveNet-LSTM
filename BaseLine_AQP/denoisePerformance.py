from data_preparation import Data_Split
from BaseLine_AQP.BaseLine import *
import pandas as pd
from evaluation_metrics import *
import warnings
from seed_set import set_seed
import global_constants as gc


warnings.filterwarnings('ignore')

indicators = ["AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AT", "ATM", "WDIR", "WFS", "PRCP", "RH"]

predict_step= gc.STEP()
poses = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
EPOSCHS = 100
BATCH_SIZE = 150
window_size = 3

for p in poses:
    X_pos = poses[p]
    Y_pos = X_pos
    df_empty = pd.DataFrame(columns=['filter1', 'BPNN_MAPE', 'LSTM_MAPE', 'RNN_MAPE','filter2', 'BPNN_SMAPE', 'LSTM_SMAPE', 'RNN_SMAPE'])

    for filter_size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        # 数据预处理
        set_seed(1)
        HIDDEN_SIZE= int((window_size + predict_step) * 2 / 3)

        dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=window_size, predict_step=1, filter_size=filter_size)
        train_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE, shuffle=True)
        dataset.X_test_scaler=torch.from_numpy(dataset.X_test_scaler) # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。

        bp=BPNet(window_size=window_size, hidden_size=HIDDEN_SIZE, predict_step=1)
        rnn=RNN(hidden_size=HIDDEN_SIZE,predict_step=1)
        lstm=LSTM(hidden_size=HIDDEN_SIZE,predict_step=1)

        optimizer_bp = torch.optim.Adam(bp.parameters(),lr=0.01)
        optimizer_rnn = torch.optim.Adam(rnn.parameters(),lr=0.01)
        optimizer_lstm = torch.optim.Adam(lstm.parameters(),lr=0.01)
        loss_bp_func=nn.MSELoss()
        loss_rnn_func=nn.MSELoss()
        loss_lstm_func=nn.MSELoss()
        # training and testing
        for epoch in range(EPOSCHS):
            for step,(b_x,b_y) in enumerate(train_loader):
                b_x=torch.tensor(b_x,dtype=torch.float32)
                b_y=torch.tensor(b_y,dtype=torch.float32)

                out_bp = bp(b_x)
                loss_bp=loss_bp_func(out_bp, b_y)
                optimizer_bp.zero_grad()

                out_rnn = rnn(b_x)
                loss_rnn=loss_rnn_func(out_rnn, b_y)
                optimizer_rnn.zero_grad()

                out_lstm = lstm(b_x)
                loss_lstm=loss_rnn_func(out_lstm, b_y)
                optimizer_lstm.zero_grad()

                loss_bp.backward()
                loss_rnn.backward()
                loss_lstm.backward()

                optimizer_bp.step()
                optimizer_rnn.step()
                optimizer_lstm.step()
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
        print("filer={}".format(filter_size), "| bp_mse:",bp_mse ,"| bp_mape: ",bp_mape, "| bp_rmse: ",bp_rmse,"| bp_mae: ",bp_mae,"| bp_r2: ",bp_r2, "| bp_smape: ",bp_smape)

        rnn_mae, rnn_mse, rnn_mape, rnn_smape, rnn_rmse, rnn_r2 = evaluate(y_true, rnn_y_pred)
        print("filer={}".format(filter_size), "| rnn_mse:", rnn_mse, "| rnn_mape: ", rnn_mape, "| rnn_rmse: ", rnn_rmse, "| rnn_mae: ", rnn_mae, "| rnn_r2: ", rnn_r2, "| rnn_smape: ", rnn_smape)

        lstm_mae, lstm_mse, lstm_mape, lstm_smape, lstm_rmse, lstm_r2 = evaluate(y_true, lstm_y_pred)
        print("filer={}".format(filter_size), "| lstm_mse:", lstm_mse, "| lstm_mape: ", lstm_mape, "| lstm_rmse: ", lstm_rmse, "| lstm_mae: ", lstm_mae, "| lstm_r2: ",lstm_r2, "| lstm_smape: ",lstm_smape)

        df_empty=df_empty.append([{'filter1': filter_size,'BPNN_MAPE':bp_mape,'RNN_MAPE':rnn_mape,'LSTM_MAPE':lstm_mape,'filter2': filter_size,'BPNN_SMAPE': bp_smape,'RNN_SMAPE': rnn_smape,'LSTM_SMAPE': lstm_smape}])

    df_empty.to_csv("denoise_performance_{}.csv".format(indicators[X_pos]))
