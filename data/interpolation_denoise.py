import pandas as pd
import numpy as np

df = pd.read_csv(r"raw_202105.csv")

#先用线性插值进行缺失值补全
results = df.interpolate(method='linear', limit_direction='both')

#删除异常值
for index, column in results.iteritems():
    if index=='Time':
        continue
    m = np.mean(column)
    s = np.std(column)

    rowNum = 0
    for item in column:
        if item>m+3*s or item<m-3*s:
            df.loc[rowNum, index] = np.nan
        rowNum = rowNum +1

# 再次用线性插值进行缺失值补全
results = df.interpolate(method='linear', limit_direction='both')

#结果保存
infilled_file = "./lt-filled-denoise-filled_202105.csv"

results.to_csv(infilled_file, index=False)