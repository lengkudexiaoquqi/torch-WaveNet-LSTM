import os
import warnings
import pandas as pd

from BaseLine_AQP.BaseLine import *
from data_preparation import Data_Split
from seed_set import set_seed
import global_constants as gc

warnings.filterwarnings('ignore')
from evaluation_metrics import *

set_seed(1)
current_dir = os.getcwd()

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

    df_bp = pd.DataFrame(columns=['window_size', 'MAPE', 'RMSE', 'R2', 'MAE', 'SMAPE'])

    for ws in window_size:
        set_seed(1)
        # 设置超参数
        HIDDEN_SIZE = int((ws + predict_step) * 2 / 3)

        dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=ws, predict_step=1,
                             filter_size=gc.SSA_FILTER_COMPONENT(p))
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

        lstm = LSTM(hidden_size=HIDDEN_SIZE, predict_step=1)

        optimizer_bp = torch.optim.Adam(lstm.parameters(), lr=LR)
        loss_func = nn.MSELoss()
        # training and testing
        for epoch in range(EPOSCHS):
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = torch.as_tensor(b_x, dtype=torch.float32)
                b_y = torch.as_tensor(b_y, dtype=torch.float32)

                outout = lstm(b_x)
                loss = loss_func(outout, b_y)
                optimizer_bp.zero_grad()
                loss.backward()
                optimizer_bp.step()
        dataset.X_test_scaler = torch.as_tensor(dataset.X_test_scaler, dtype=torch.float32)
        y_true = dataset.y_test

        test_output_bp = lstm(dataset.X_test_scaler)
        lstm_y_pred = test_output_bp.detach().numpy()
        # 反归一化
        lstm_y_pred = dataset.scaler.inverse_transform(lstm_y_pred)

        lstm_mae, lstm_mse, lstm_mape, lstm_smape, lstm_rmse, lstm_r2 = evaluate(y_true, lstm_y_pred)
        print(indicators[p], ": current_window_size：", ws, "| lstm_mse:", lstm_mse, "| lstm_mape: ", lstm_mape, "| lstm_rmse: ", lstm_rmse, "| lstm_mae: ", lstm_mae, "| lstm_r2: ", lstm_r2, "| lstm_smape: ", lstm_smape)

        df_bp = df_bp.append([{'window_size': ws,'MAPE': lstm_mape,'SMAPE': lstm_smape,'RMSE': lstm_rmse,'R2': lstm_r2, 'MAE': lstm_mae}])

    path = current_dir + "/{}/".format(indicators[p]) + "LSTM_{}.csv".format(LR)
    df_bp.to_csv(path, encoding='utf-8')
