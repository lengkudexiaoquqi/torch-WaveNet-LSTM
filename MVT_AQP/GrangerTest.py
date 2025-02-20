from statsmodels.tsa import stattools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# 读取数据
df1 = pd.read_csv(r"..\data\lt-filled-denoise-filled.csv", index_col=0)
df1= df1 [["AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AT", "ATM", "WDIR", "WFS", "PRCP", "RH"]]
#
# df2 = pd.read_csv("../data/res_last_4_12.csv",index_col=0)
# df1["风向"]=df2["风向"].values
# df1["降水量"]=df2["降水量"].values


window_size=dict({
    "AQI": 8,
    "PM2.5": 4,
    "PM10": 8,
    "NO2": 8,
    "SO2":8,
    "O3": 8,
    "CO":8
})

adfResults = pd.DataFrame(np.zeros([13, 6]), columns=["Indicator", "ADF Statistic", "p-value", "1%", "5%", "10%"])
count = 0
# 单方根检验（平稳性检验）
for k in df1.columns:
    adfResults.loc[count, 'Indicator'] = k
    res=stattools.adfuller(df1[k])
    adfResults.loc[count, 'ADF Statistic'] = res[0]
    adfResults.loc[count, 'p-value'] = res[1]
    for key, value in res[4].items():
        adfResults.loc[count, key] = value
    count = count + 1
adfResults.to_csv("./adfResults-2020.csv")

meteorologicalList= ["AT", "ATM", "WDIR", "WFS", "PRCP", "RH"]
grangercausalitytestResults = pd.DataFrame(np.zeros([7, 6]), index=["AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO"], columns=meteorologicalList)
# 格兰杰因果检验
for key, value in window_size.items():
    for m in meteorologicalList:
        res1=stattools.grangercausalitytests(df1[[key, m]], maxlag=value)
        fTest = res1[value]
        print(fTest[0]['ssr_ftest'][1])
        grangercausalitytestResults.loc[key, m] = fTest[0]['ssr_ftest'][1]
grangercausalitytestResults.to_csv("./grangercausalitytestResults-2020.csv")
