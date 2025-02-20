import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class Data_Split(Dataset):
    def __init__(self, x_pos, y_pos, window_size, predict_step, filter_size, feature=1):
        '''
        :param x_pos: 输入的x下标位置
        :param y_pos: 输出的y下标位置
        :param window_size: 滑动窗口大小
        :param predict_step: 预测步长
        '''
        super(Data_Split, self).__init__()
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.window_size = window_size
        self.predict_step = predict_step
        self.feature = feature
        self.original_df = pd.read_csv(r"..\data\lt-filled-denoise-filled.csv", index_col=0)
        # self.df = self.df[["AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AT", "ATM", "WDIR", "WFS", "PRCP", "RH"]]
        self.filtered_df = pd.read_csv(r"..\data\ssa{}.csv".format(filter_size), header=None)
        self.X_train_scaler, self.y_train_scaler, self.X_test_scaler, self.y_test, self.scaler = self.slide_window()

    def __getitem__(self, index):
        X = self.X_train_scaler[index]
        Y = self.y_train_scaler[index]
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        return X, Y

    def slide_window(self):
        data_train = self.original_df.iloc[:int(self.original_df.shape[0] * 0.8), :].values
        data_test = self.original_df.iloc[int(self.original_df.shape[0] * 0.8):, :].values
        data_train_filter = self.filtered_df.iloc[:int(self.filtered_df.shape[0] * 0.8), :].values
        data_test_filter = self.filtered_df.iloc[int(self.filtered_df.shape[0] * 0.8):, :].values
        data_train.astype(np.float32)
        data_test.astype(np.float32)
        data_test_filter.astype(np.float32)
        data_train_filter.astype(np.float32)
        X_train, y_train, X_test, y_test = self.data_split(data_train, data_test)
        X_train_filter, y_train_filter, X_test_filter, y_test_filter = self.data_split(data_train_filter,
                                                                                        data_test_filter)

        y_test = np.reshape(y_test, (y_test.shape[0], -1))

        X_train_filter = np.reshape(X_train_filter, (X_train_filter.shape[0], -1))  # 若window_size=3，则将形如[[1,2,3],[4,5,6],...,[97,98,99]]的输入转换为[[1 2 3], [4 5 6], ...,[97 98 99]]，下同
        y_train_filter = np.reshape(y_train_filter, (X_train_filter.shape[0], -1))
        X_test_filter = np.reshape(X_test_filter, (X_test_filter.shape[0], -1))
        y_test_filter = np.reshape(y_test_filter, (y_test_filter.shape[0], -1))
        # 归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaler = scaler.fit_transform(X_train_filter)
        X_test_scler = scaler.transform(X_test_filter)
        X_train_scaler = np.reshape(X_train_scaler, (X_train_scaler.shape[0], -1, self.feature))  # 52行的反操作
        X_test_scaler = np.reshape(X_test_scler, (X_test_scler.shape[0], -1, self.feature))
        # scaler1 = MinMaxScaler(feature_range=(0.001, 1))
        y_train_scaler = scaler.fit_transform(y_train_filter)
        return X_train_scaler, y_train_scaler, X_test_scaler, y_test, scaler

    def __len__(self):
        return int(self.filtered_df.shape[0] * 0.8) - self.window_size + 1 - self.predict_step

    def data_split(self, data_train, data_test):
        X_train = np.array(
            [data_train[i: i + self.window_size, self.x_pos] for i in
             range(data_train.shape[0] - self.window_size - self.predict_step + 1)])
        y_train = np.array(
            [data_train[i + self.window_size:i + self.window_size + self.predict_step, self.y_pos] for i in
             range(data_train.shape[0] - self.window_size - self.predict_step + 1)])
        X_test = np.array(
            [data_test[i: i + self.window_size, self.x_pos] for i in
             range(data_test.shape[0] - self.window_size - self.predict_step + 1)])
        y_test = np.array([data_test[i + self.window_size:i + self.window_size + self.predict_step, self.y_pos] for i in
                           range(data_test.shape[0] - self.window_size - self.predict_step + 1)])

        # 让X_train和x_test多了一维，不知道为什么？
        if isinstance(self.x_pos, int) or (isinstance(self.x_pos, list) and len(self.x_pos) == 1):
            X_train = np.expand_dims(X_train, axis=2)
            X_test = np.expand_dims(X_test, axis=2)
        return X_train, y_train, X_test, y_test
