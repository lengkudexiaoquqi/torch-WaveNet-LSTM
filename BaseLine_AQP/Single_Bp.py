from sklearn.preprocessing import MinMaxScaler
from data_preparation import Data_Split
from BaseLine_AQP.BaseLine import *
import pandas as pd
from sklearn.metrics import *
from torchsummary import summary
import numpy as np
import warnings
import torch
from seed_set import set_seed

warnings.filterwarnings('ignore')
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

df=pd.read_csv(r"../data/fill_last.csv",index_col=0)
df=df[["AQI","PM2.5","PM10","SO2","NO2","O3","CO","温度","压强","风力","降水量","湿度"]]
predict_step=1
window_size = [14]
for ws in window_size:
    # 数据预处理
    set_seed(1)
    X_pos=0
    Y_pos=0

    # 设置超参数
    HIDDEN_SIZE= int((ws+predict_step)*2/3)
    # HIDDEN_SIZE= np.ceil(np.log2(ws)) if ws>1 else 1
    # HIDDEN_SIZE= 32
    EPOSCHS=1000
    BATCH_SIZE=150
    LR=0.01

    dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=ws, predict_step=1, filter_size=40)
    train_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE, shuffle=False)
    dataset.X_test_scaler=torch.from_numpy(dataset.X_test_scaler)

    bp=BPNet(window_size=ws,hidden_size=HIDDEN_SIZE,predict_step=1)

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
        test_output = bp(dataset.X_test_scaler)
        y_pred = test_output.detach().numpy()
        # 反归一化
        y_pred = dataset.scaler1.inverse_transform(y_pred)
        y_true = dataset.y_test
        # mae=mean_absolute_error(y_true,y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mape = MAPE(y_true, y_pred)

    print("*"*100)
    print('| test loss: %.4f' % mse,'| test mape: %.4f' % mape)








