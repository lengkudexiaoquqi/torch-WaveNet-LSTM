import os
import warnings
import pandas as pd

import global_constants as gc
from BaseLine_AQP.BaseLine import *
from evaluation_metrics import *
from data_preparation import Data_Split
from seed_set import set_seed

current_dir = os.getcwd()
warnings.filterwarnings('ignore')

indicators = gc.INDICATORS()
poses = gc.POSES()
predict_step = gc.STEP()
window_size = gc.WINDOW_SIZES()
EPOSCHS = 1000
BATCH_SIZE = 150
LR = 0.01

for p in poses:
    X_pos = poses[p]
    Y_pos = X_pos
    df_rnn = pd.DataFrame(columns=['window_size', 'MAPE', 'RMSE', 'R2', 'MAE', 'SMAPE'])
    for ws in window_size:
        set_seed(1)

        HIDDEN_SIZE = int((ws + predict_step) * 2 / 3)

        dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=ws, predict_step=predict_step, filter_size=gc.SSA_FILTER_COMPONENT(p))
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

        rnn = RNN(hidden_size=HIDDEN_SIZE, predict_step=1)

        optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=LR)
        loss_func = nn.MSELoss()
        # training and testing
        for epoch in range(EPOSCHS):
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = torch.as_tensor(b_x, dtype=torch.float32)
                b_y = torch.as_tensor(b_y, dtype=torch.float32)

                outout = rnn(b_x)
                loss = loss_func(outout, b_y)
                optimizer_rnn.zero_grad()
                loss.backward()
                optimizer_rnn.step()

        dataset.X_test_scaler = torch.as_tensor(dataset.X_test_scaler, dtype=torch.float32)
        y_true = dataset.y_test

        test_output_rnn = rnn(dataset.X_test_scaler)
        rnn_y_pred = test_output_rnn.detach().numpy()
        # 反归一化
        rnn_y_pred = dataset.scaler.inverse_transform(rnn_y_pred)

        rnn_mae, rnn_mse, rnn_mape, rnn_smape, rnn_rmse, rnn_r2 = evaluate(y_true, rnn_y_pred)
        print(indicators[p], ":current_window_size：", ws, "| rnn_mse:", rnn_mse, "| rnn_mape: ", rnn_mape, "| rnn_rmse: ", rnn_rmse, "| rnn_mae: ", rnn_mae, "| rnn_r2: ", rnn_r2, "| rnn_smape: ", rnn_smape)
        df_rnn = df_rnn.append([{'window_size': ws, 'MAPE': rnn_mape, 'SMAPE': rnn_smape, 'RMSE': rnn_rmse, 'R2': rnn_r2, 'MAE': rnn_mae}])

    path = current_dir + "/{}/".format(indicators[p]) + "RNN_{}.csv".format(LR)
    df_rnn.to_csv(path, encoding='utf-8')
