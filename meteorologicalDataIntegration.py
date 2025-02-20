import pandas as pd
import os
import re
import datetime
dir_path=r"2021"
combine=[]
file_names=os.listdir(dir_path)


for file_name in file_names:
    res=re.match(r'.*?\_(.*?)\.csv',file_name)
    date_time=res.group(1)
    # print(type(date_time))
    abstract_path=dir_path+r"\\"+file_name
    f=open(abstract_path,encoding='utf-8')
    df=pd.read_csv(f,encoding='utf-8')
    df = df.iloc[0:-1]
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df["hour"] = df["hour"].apply(str)
    print(date_time)
    year=int(date_time[:4])
    month=int(date_time[4:6])
    day=int(date_time[6:8])
    current_time = datetime.datetime(year,month,day)
    print(current_time.strftime("%y/%m/%d"))
    pre_time = current_time - datetime.timedelta(days=1)
    index = int(df[df["hour"] == '0'].index[0])
    df["hour"][:index + 1] = df["hour"][:index + 1].apply(lambda x: current_time.strftime('%Y/%m/%d') + " " + str(x) + ":00:00")
    df["hour"][index + 1:] = df["hour"][index + 1:].apply(lambda x: pre_time.strftime('%Y/%m/%d') + " " + str(x) + ":00:00")
    print(df)
    combine.append(df)
    f.close()


last_Res=pd.concat(combine)
last_Res["hour"]=pd.to_datetime(last_Res["hour"])
last_Res=last_Res.drop_duplicates()
last_Res.index=last_Res['hour']
last_Res.sort_index(inplace=True)
last_Res.drop_duplicates(inplace=True)

last_Res.to_csv("meteorological_res.csv",encoding='utf-8-sig')



