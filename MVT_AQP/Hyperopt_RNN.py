import os
import warnings

import joblib
import numpy as np
import optuna
import torch.utils.data
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.metrics import mean_squared_error

import global_constants as gc
from MVT_AQP.BaseLine_MVT import *
from data_preparation import Data_Split
from evaluation_metrics import *
from logger_res import get_logger, get_current_time
from seed_set import set_seed
from evaluation_metrics import *

warnings.filterwarnings('ignore')


DEVICE = torch.device("cpu")
DIR = os.getcwd()
set_seed(1)

def objective(trial):
    set_seed(1)
    global X_pos
    global Y_pos
    global Window_size
    HIDDEN_SIZE = trial.suggest_int("hidden_size",low=int((Window_size + 1) * 2 / 3), high=int((Window_size + 1) * 2 / 3)*2)
    EPOSCHS = 1000
    BATCH_SIZE = 150

    LR = trial.suggest_float("learning_rate", 0.001, 0.01, log=True)
    FEATURE=len(X_pos)
    dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=Window_size, predict_step=gc.STEP(), filter_size=gc.SSA_FILTER_COMPONENT(Y_pos), feature=FEATURE)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)


    rnn = RNN(hidden_size=HIDDEN_SIZE,predict_step=1,input_dim=FEATURE)
    optimizer_rnn = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    # training and testing
    for epoch in range(EPOSCHS):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = torch.as_tensor(b_x, dtype=torch.float32)
            b_y = torch.as_tensor(b_y, dtype=torch.float32)

            outout = rnn(b_x)
            loss = loss_func(outout, b_y)
            optimizer_rnn.zero_grad()
            loss.backward()
            optimizer_rnn.step()

    dataset.X_test_scaler = torch.as_tensor(dataset.X_test_scaler, dtype=torch.float32)
    test_output = rnn(dataset.X_test_scaler)
    y_pred = test_output.detach().numpy()
    # 反归一化
    y_pred = dataset.scaler.inverse_transform(y_pred)
    y_true = dataset.y_test
    mae, mse, mape, smape, rmse, r2 = evaluate(y_true, y_pred)
    logger.info("current_hidden_num----> {0}   current_lr------>{1}".format(HIDDEN_SIZE,LR))
    logger.info("mae={0:.4f},mse={1:.4f},r2={2:.4f},rmse={3:.4f},mape={4:.4f},smape={4:.4f}".format(mae, mse, r2, rmse, mape, smape))
    logger.info("-------------------------------------------------")
    trial.report(mape, epoch)
    base_path = '../MVT_AQP/PM2.5_湿度/rnn/'
    file_name_model=base_path+"mape_{0:.4f}_{1}.pkl".format(mape,current_time)
    file_name_path =base_path+"mape_{0:.4f}_{1}.npy".format(mape,current_time)
    torch.save(rnn,file_name_model)
    np.save(file_name_path,y_pred)
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mape


if __name__ == "__main__":
    set_seed(1)
    # 让结果可复现
    sampler = TPESampler(seed=1)
    current_time = get_current_time()

    indicators = gc.INDICATORS()
    poses = gc.POSES()
    predict_step = gc.STEP()

    for p in poses:
        Y_pos = p
        X_pos = gc.CAUSAL_FACTORS(Y_pos)
        Window_size = gc.OPTIMAL_WINDOW_SIZE(Y_pos)
        current_predict_column = indicators[Y_pos]
        filename = '../MVT_AQP/{0}/rnn/{0}_{1}.log'.format(current_predict_column, current_time)
        logger = get_logger(filename)

        combine_name = ''
        for i in X_pos:
            combine_name += indicators[i]
            combine_name += '_'
        study = optuna.create_study(study_name='rnn',direction="minimize",sampler=sampler)
        study.optimize(objective, n_trials=100)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        logger.info("Study statistics: ")
        logger.info("Number of finished trials:{}".format(len(study.trials)))
        logger.info("Number of pruned trials:{}".format(len(pruned_trials)))
        logger.info("Number of complete trials:{}".format(len(complete_trials)))
        logger.info("Best trial:")

        trial = study.best_trial

        logger.info("  Value:{}".format(trial.value))

        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))

        df=study.trials_dataframe()
        csv_name ="hyperopt_rnn_{0}.csv".format(combine_name)
        df.to_csv(csv_name)
        # 保存和恢复study
        # 可视化
        optuna.visualization.plot_slice(study, params=['hidden_size', 'learning_rate'])
        joblib.dump(study,"study_rnn_{0}.pkl".format(combine_name))
        # 恢复study
        # study= joblib.load('study.pkl')
