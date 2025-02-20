import os
import warnings

import pandas as pd

from BaseLine_AQP.BaseLine import *
from evaluation_metrics import *
from data_preparation import Data_Split
from seed_set import set_seed
import global_constants as gc

set_seed(1)
# python get current dir\
current_dir = os.getcwd()

indicators = gc.INDICATORS()
poses = gc.POSES()
predict_step = gc.STEP()
window_size = gc.WINDOW_SIZES()
EPOSCHS = 1000
BATCH_SIZE = 150
LR = 0.01

for p in poses:
    X_pos = p
    Y_pos = X_pos

    df_bp = pd.DataFrame(columns=['window_size', 'MAPE', 'RMSE', 'R2', 'MAE', 'SMAPE'])

    for ws in window_size:
        # 数据预处理
        set_seed(1)
        # 设置超参数
        HIDDEN_SIZE = int((ws + predict_step) * 2 / 3)

        dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=ws, predict_step=predict_step,
                             filter_size=gc.SSA_FILTER_COMPONENT(X_pos))
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

        bp = BPNet(window_size=ws, hidden_size=HIDDEN_SIZE, predict_step=1)

        optimizer_bp = torch.optim.Adam(bp.parameters(), lr=LR)
        loss_func = nn.MSELoss()
        # training and testing
        for epoch in range(EPOSCHS):
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = torch.as_tensor(b_x, dtype=torch.float32)
                b_y = torch.as_tensor(b_y, dtype=torch.float32)

                outout = bp(b_x)
                loss = loss_func(outout, b_y)
                optimizer_bp.zero_grad()
                loss.backward()
                optimizer_bp.step()
        dataset.X_test_scaler = torch.as_tensor(dataset.X_test_scaler, dtype=torch.float32)
        y_true = dataset.y_test

        test_output_bp = bp(dataset.X_test_scaler)
        bp_y_pred = test_output_bp.detach().numpy()
        # 反归一化
        bp_y_pred = dataset.scaler.inverse_transform(bp_y_pred)

        bp_mae, bp_mse, bp_mape, bp_smape, bp_rmse, bp_r2 = evaluate(y_true, bp_y_pred)
        print("current_window_size：", ws, "| bp_mse:", bp_mse, "| bp_mape: ", bp_mape, "| bp_rmse: ", bp_rmse,
              "| bp_mae: ", bp_mae, "| bp_r2",
              bp_r2, "| bp_smape", bp_smape)
        df_bp = df_bp.append(
            [{'window_size': ws, 'MAPE': bp_mape, 'RMSE': bp_rmse, 'R2': bp_r2, 'MAE': bp_mae, 'SMAPE': bp_smape}])

    path = current_dir + "/{}/".format(indicators[p]) + "BPNN_{}.csv".format(LR)
    df_bp.to_csv(path, encoding='utf-8')
