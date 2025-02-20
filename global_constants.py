def SSA_FILTER_COMPONENT(index):
    dic = {0:40, 1: 30, 2: 40, 3: 40, 4: 40, 5: 30, 6: 30}
    return dic[index]

# ["AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AT", "ATM", "WDIR", "WFS", "PRCP", "RH"]
def OPTIMAL_WINDOW_SIZE(index):
    dic = {0:8, 1: 8, 2: 8, 3: 8, 4: 4, 5: 8, 6: 8}
    return dic[index]

def CAUSAL_FACTORS(index):
    dic = {0: [0, 12], 1: [1, 7], 2: [2, 7], 3: [3, 7], 4: [4, 7], 5: [5, 7], 6: [6, 7]}
    return dic[index]

def INDICATORS():
    return ["AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AT", "ATM", "WDIR", "WFS", "PRCP", "RH"]

def POSES():
    return [0, 1, 2, 3, 4, 5, 6]

def STEP():
    return 1

def WINDOW_SIZES():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]