import warnings

import geatpy as ea
import torch.utils.data

import global_constants as gc
from WaveNet_LSTM_AQP.WaveNet_LSTM import *
from data_preparation import Data_Split
from evaluation_metrics import *
from seed_set import set_seed

class MyProblem(ea.Problem):
    def __init__(self):
        name = 'DE-waveNet-lstm'
        M = 1  # 目标维数,针对单一的指标寻优，所以只是一维
        maxormins = [1]  # 初始化目标最小最大化标记列表， 1： min； -1： max
        Dim = 5  # 决策变量的维数,skip_filter,res_filter,dilated,lstm_units,lr,
        varTypes = [1,1,1,1,0]  # 0 连续  1 离散
        lb = [24,24,1,1,0.0001]  # 决策变量下界
        ub = [256,256,3,64,0.03]  # 决策变量的上边界
        lbin = [1,1,1,1,1]  # 决策变量的下边界
        ubin = [1,1,1,1,1]  # 决策变量的上边界

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self,pop):  # 目标函数，pop为传入的种群对象
        # 解码种群得到变量
        Vars = pop.Phen  # 得到决策变量矩阵
        SKIP_FILTERS = Vars[:, [0]]
        RES_FILTERS = Vars[:, [1]]
        DILATEDS = Vars[:, [2]]
        LSTM_UNITS=Vars[:,[3]]
        LRS=Vars[:,[4]]
        res = []
        for i in range(Vars.shape[0]):
            set_seed(1)
            SKIP_FILTER = int(SKIP_FILTERS[i, :][0])
            RES_FILTER = int(RES_FILTERS[i, :][0])
            DILATED = int(DILATEDS[i, :][0])
            LSTM_UNIT=int(LSTM_UNITS[i, :][0])
            LR = round(LRS[i, :][0],3)


            EPOSCHS = 1000
            BATCH_SIZE= 150
            Y_pos = 0
            X_pos = gc.CAUSAL_FACTORS(Y_pos)
            Window_size = gc.OPTIMAL_WINDOW_SIZE(Y_pos)
            FEATURE = len(X_pos)

            dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=Window_size, predict_step=1, filter_size=50,
                                 feature=FEATURE)
            train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
            dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

            lstm = WaveNet_LSTM(input_size=FEATURE, residual_size=RES_FILTER, skip_size=SKIP_FILTER,
                                dilation_depth=DILATED, hidden_size=LSTM_UNIT)
            # lstm.to(device)

            # lstm.to(device)
            weight_p, bias_p = [], []
            for name, p in lstm.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            # optimizer_lstm = torch.optim.Adam([{'params': weight_p, 'weight_decay': 0},
            #                                    {'params': bias_p, 'weight_decay': 0}],
            #                                   lr=LR)
            optimizer_lstm = torch.optim.Adam(lstm.parameters(),LR)
            loss_func = nn.MSELoss()
            # training and testing
            for epoch in range(EPOSCHS):
                for step, (b_x, b_y) in enumerate(train_loader):
                    b_x = torch.as_tensor(b_x, dtype=torch.float32)
                    b_y = torch.as_tensor(b_y, dtype=torch.float32)

                    outout = lstm(b_x)
                    loss = loss_func(outout, b_y)
                    optimizer_lstm.zero_grad()
                    loss.backward()
                    optimizer_lstm.step()
                    # if epoch % 50 == 0:
                    #     print({"epoch": epoch, "loss": loss.item()})
            # lstm.to(device='cpu')
            dataset.X_test_scaler = torch.as_tensor(dataset.X_test_scaler, dtype=torch.float32)
            test_output = lstm(dataset.X_test_scaler)
            # print("正在计算")
            y_pred = test_output.detach().numpy()
            # 反归一化
            y_pred = dataset.scaler.inverse_transform(y_pred)
            y_true = dataset.y_test
            mae, mse, mape, smape, rmse, r2 = evaluate(y_true, y_pred)
            print("mae={0:.4f},mse={1:.4f},r2={2:.4f},rmse={3:.4f},mape={4:.4f},smape={4:.4f}".format(mae, mse, r2, rmse, mape, smape))
            print("-------------------------------------------------")
            res.append(mape)
        # 计算结果
        pop.ObjV = np.reshape(np.array(res), (-1, 1))


if __name__ == '__main__':
    set_seed(1)
    problem = MyProblem()
    Encoding = 'RI'  # 编码方式
    NIND = 20   # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    #     实例化一个算法模板对象
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
    myAlgorithm.MAXGEN = 15
    myAlgorithm.mutOper.F = 0.5
    myAlgorithm.recOper.XOVR = 0.7
    myAlgorithm.logTras = 1
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 1
    [BestIndi, population] = myAlgorithm.run()
    BestIndi.save()
    param_dict={
        0: "SKIP_FILTER",
        1: "RES_FILTER",
        2: "DILATED",
        3: "LSTM_UNIT",
        4: "LR"
    }
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为： %s' % BestIndi.ObjV[0][0])
        print('最优的控制变量值为:')
        for i in range(BestIndi.Phen.shape[1]):
            print("参数： {0}----------->{1}".format(param_dict.get(i),BestIndi.Phen[0, i]))




