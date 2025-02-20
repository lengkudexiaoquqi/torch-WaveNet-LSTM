from dm_test import *
import numpy as np
from sklearn.metrics import *
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


pos= 2
constants={0: 'AQI', 1: 'PM2.5', 2: 'PM10', 3: 'SO2', 4: 'NO2', 5: 'O3', 6: 'CO', 7: '温度', 8: '压强', 9: '风力', 10: '降水量', 11: '湿度'}


# 读取三个文件
muti_bp=np.load("muti_factor_predict_value/bp/{}.npy".format(constants.get(pos)))
temp_bp=np.load("muti_factor_predict_value/bp/{}.npy".format(constants.get(pos)))
muti_bp=np.reshape(muti_bp,(1,-1))
muti_bp=muti_bp.tolist()


muti_rnn=np.load("muti_factor_predict_value/rnn/{}.npy".format(constants.get(pos)))
temp_rnn=np.load("muti_factor_predict_value/rnn/{}.npy".format(constants.get(pos)))

# muti_rnn=np.load("muti_factor_predict_value/rnn/O3.npy")
# temp_rnn=np.load("muti_factor_predict_value/rnn/O3.npy")
muti_rnn=np.reshape(muti_rnn,(1,-1))
muti_rnn=muti_rnn.tolist()



muti_lstm=np.load("muti_factor_predict_value/lstm/{}.npy".format(constants.get(pos)))

temp_lstm=np.load("muti_factor_predict_value/lstm/{}.npy".format(constants.get(pos)))

# muti_lstm=np.load("muti_factor_predict_value/lstm/O3.npy")
# temp_lstm=np.load("muti_factor_predict_value/lstm/O3.npy")
muti_lstm=np.reshape(muti_lstm,(1,-1))
muti_lstm=muti_lstm.tolist()


wavenet=np.load("WaveNet-LSTM/{}.npy".format(constants.get(pos)))
temp_wavenet=np.load("WaveNet-LSTM/{}.npy".format(constants.get(pos)))

wavenet=np.reshape(wavenet,(1,-1))
wavenet=wavenet.tolist()


real_data=np.load("real_value/{}.npy".format(constants.get(pos)))

y_true=np.load("real_value/{}.npy".format(constants.get(pos)))
real_data=np.reshape(real_data,(1,-1))
real_data=real_data.tolist()


for y_pred in [temp_bp,temp_rnn,temp_lstm,temp_wavenet]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    rmse = RMSE(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    print("mae={0:.4f},mse={1:.4f},r2={2:.4f},rmse={3:.4f},mape={4:.4f}".format(mae, mse, r2, rmse, mape))


# # 计算单变量和多变量BP之间的DM_test
print(constants.get(pos))
print("融合模型和多变量BP")
res=dm_test(real_data[0],wavenet[0],muti_bp[0],h=1,crit="poly",power=1)
# res=dm_test(real_data[0],wavenet[0],muti_bp[0],h=1,crit="poly",power=2)
# res=dm_test(real_data[0],wavenet[0],muti_bp[0],h=1,crit="MAPE")
# res=dm_test(real_data[0],wavenet[0],muti_bp[0],h=1,crit="MSE")
# res=dm_test(real_data[0],wavenet[0],muti_bp[0],h=1,crit="MAD")
print(res)


# 计算单变量和多变量BP之间的DM_test
print(constants.get(pos))
print("融合模型和多变量RNN")
res=dm_test(real_data[0],wavenet[0],muti_rnn[0],h=1,crit="poly",power=1)
# res=dm_test(real_data[0],wavenet[0],muti_rnn[0],h=1,crit="poly",power=2)
# res=dm_test(real_data[0],wavenet[0],muti_rnn[0],h=1,crit="MAPE")
# res=dm_test(real_data[0],wavenet[0],muti_rnn[0],h=1,crit="MSE")
# res=dm_test(real_data[0],wavenet[0],muti_rnn[0],h=1,crit="MAD")
print(res)


# 计算单变量和多变量BP之间的DM_test
print(constants.get(pos))
print("融合模型和多变量LSTM")
res=dm_test(real_data[0],wavenet[0],muti_lstm[0],h=1,crit="poly",power=1)
# res=dm_test(real_data[0],wavenet[0],muti_lstm[0],h=1,crit="poly",power=2)
# res=dm_test(real_data[0],wavenet[0],muti_lstm[0],h=1,crit="MAPE")
# res=dm_test(real_data[0],wavenet[0],muti_lstm[0],h=1,crit="MSE")
# res=dm_test(real_data[0],wavenet[0],muti_lstm[0],h=1,crit="MAD")
print(res)