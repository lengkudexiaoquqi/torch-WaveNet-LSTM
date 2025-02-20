from sklearn.preprocessing import MinMaxScaler
from data_preparation import Data_Split
from BaseLine_AQP.BaseLine import *
import pandas as pd
from sklearn.metrics import *
from torchsummary import summary
import numpy as np
import warnings

from evaluation_metrics import evaluate
from seed_set import set_seed
import os
import global_constants as gc


warnings.filterwarnings('ignore')
set_seed(1)
# python get current dir\
current_dir= os.getcwd()


indicators = gc.INDICATORS()
poses = [0]
predict_step = gc.STEP()
window_size = gc.WINDOW_SIZES()
EPOSCHS = 500
BATCH_SIZE = 150
LR = 0.01

for p in poses:
    X_pos = 0
    Y_pos = 0

    df_bp = pd.DataFrame(columns=['window_size', 'MAPE', 'RMSE', 'R2', 'MAE', 'SMAPE'])


    for ws in window_size:
        set_seed(1)
        print(gc.SSA_FILTER_COMPONENT(X_pos))
        # 设置超参数
        HIDDEN_SIZE = int((ws + predict_step) * 2 / 3)

        dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=ws, predict_step=1,
                             filter_size=40)
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

        bp = BPNet(window_size=ws, hidden_size=HIDDEN_SIZE, predict_step=1)

        optimizer_bp = torch.optim.Adam(bp.parameters(),lr=LR)
        loss_func=nn.MSELoss()
        # training and testing
        for epoch in range(EPOSCHS):
            for step,(b_x,b_y) in enumerate(train_loader):
                b_x=torch.tensor(b_x,dtype=torch.float32)
                b_y=torch.tensor(b_y,dtype=torch.float32)
                outout = bp(b_x)
                loss=loss_func(outout,b_y)
                optimizer_bp.zero_grad()
                loss.backward()
                optimizer_bp.step()
        dataset.X_test_scaler = torch.tensor(dataset.X_test_scaler, dtype=torch.float32)
        y_true = dataset.y_test

        test_output_bp = bp(dataset.X_test_scaler)
        bp_y_pred = test_output_bp.detach().numpy()
        # 反归一化
        bp_y_pred = dataset.scaler.inverse_transform(bp_y_pred)


        # 评估三个模型的性能
        print("%" * 100)
        print("current_window_size：",ws)

        bp_mae, bp_mse, bp_mape, bp_smape, bp_rmse, bp_r2 = evaluate(y_true, bp_y_pred)
        print("current_window_size：", ws, "| bp_mse:", bp_mse, "| bp_mape: ", bp_mape, "| bp_rmse: ", bp_rmse,
              "| bp_mae: ", bp_mae, "| bp_r2",
              bp_r2, "| bp_smape", bp_smape)






