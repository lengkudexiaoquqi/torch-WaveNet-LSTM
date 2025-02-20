# import pandas as pd
# #
# # import numpy as np
# #
# # city=["Lanzhou", "Berlin", "Lanzhou", "New York", "Berlin", "New York"]
# # country = ["China", "Germany", "China", "US", "Germany", "US"]
# # population = [100, 100, 100, 100, 100, 100]
# #
# # df = pd.DataFrame({"city":city, "country": country, "population": population})
# # print("AAAAAAAAAAA:", type(df.columns))
# #
# # print("BBBBBBBBBBB:", df.columns[0])
# #
# # print(df['country'].unique())
# #
# # print(type(df.groupby(['country'], as_index=False).mean()))
# #
# #
# # dA = {"ID": [1, 2, 3], "Chinese": [100, 90, 80]}
# # dB = {'ID': [1, 2, 3], "English": [60, 70, 80]}
# # dfA = pd.DataFrame(dA, columns = ['ID', 'Chinese'])
# # dfB = pd.DataFrame(dB, columns = ['ID', "English"])
# #
# # df = pd.merge(dfA, dfB, on='ID')
# # print(df)
# #
# # ones = np.ones((2, 3), dtype = np.int_)
# # print(ones.ndim)
# # print(ones.dtype)
# # print(ones.shape)
# #
# # s = np.arange(5)
# # print(np.sqrt(s))
# # print(np.exp(s))
# #
# # data = np.random.rand(4, 5)
# # print(data>0.4)
# #
# # print(data[(data>0.4)])
# #
# # t = np.random.rand(4, 5)
# # print(t)
# # print(t[1])
# #
# # print(np.ones([2, 3]))
# #
# # s = pd.Series([2, 3, 5, 6], index = ['a', 'b', 'c', 'd'])
# # print(s['b'])
# # print('b' in s)
#
# data = pd.read_csv('co2emissions.csv', skiprows=4)
# metadata = pd.read_csv('countries_metadata.csv', encoding='utf-8')
#
# merge = pd.merge(data, metadata, on='Country Code')
# print("AAAAAAAAAAAAAAA:", type(pd.notnull(merge['Region'])))
# merge = merge[pd.notnull(merge['Region'])]
#
# print(type(merge.columns))
#
# merge = merge.drop(merge.columns[[60, 64]], axis = 1)
# print(type(merge.count()))
#
# s = merge.count()
# print(s[s==0])
# print("*"*100)
# # print(s.describe)
#
# pdData = pd.DataFrame({"No":[1, 2, 3, 4], "Name": ['Zhang San', 'Li Si', 'Wang Wu', 'Zhao Liu'], "Sex":['M', 'F', 'F', 'M']})
# # print(pdData.describe())
# # print("*"*100)
# pdData = pd.DataFrame({"No":[1.7, 1.5, 1.8], "Age":[30 ,20, 10]})
# dd = pdData.sort_values(['No'])
# print(dd.ix[-1])


from math import *

from dm_test import dm_test


def sqrt(n):
    return n

ad = [1.228837822, 2.668400552, 3.417728064, 2.239197157, 2.122559915, 0.463798715, -0.550805454, 1.182946247, -2.413298715, 0.979470609, 0.550882332, 1.227921418, -0.923506684, -0.090278956, 1.683793346, -0.610771448, 1.281044663, -0.922245829, -0.578110038, 0.768697284]
p1 = [0.902836991, 2.449267788, 3.207558081, 2.438322146, 2.775108591, 0.593161712, 0.108518596, 0.878517725, -1.165313169, 0.593719307, -0.003627286, 0.994315258, 0.519424786, 0.285099022, 0.571378593, 0.223335902, 0.327580981, 0.084688854, -0.083990871, -0.073103735]
p2 = [0.89454344, 2.321352055, 2.520815743, 1.908074955, 0.950782083, -0.61066483, -1.115450336, 1.111630867, -2.777648216, 1.51727999, 0.489767913, 1.047002047, -1.344791792, 0.001923543, 1.746568098, -1.063167761, 1.271983737, -1.289334452, -0.464421299, 0.978531379]
print(dm_test(ad, p1, p2, h = 4))


