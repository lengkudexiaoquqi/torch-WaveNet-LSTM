import os

import joblib
import optuna
import torch.utils.data
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import pandas as pd
import global_constants as gc

from evaluation_metrics import *
from WaveNet_LSTM_AQP.WaveNet_LSTM import *
from data_preparation import Data_Split
from seed_set import set_seed

DIR = os.getcwd()
set_seed(1)

def objective(trial):
    set_seed(1)
    global X_pos
    global Y_pos
    global Window_size


    RESIDUAL_SIZE = trial.suggest_int("residual_channel", low=2, high=36)
    SKIP_SIZE = trial.suggest_int("skip_channel", low=2, high=36)
    HIDDEN_SIZE = trial.suggest_int("hidden_size", low=2, high=36)
    DILATION_DEPTH = 8

    # BATCH_SIZE = trial.suggest_int('BATCH_Size', low=10, high=200)
    BATCH_SIZE=150
    # EPOSCHS = trial.suggest_int("EPOCHS", low=200, high=2000)
    # LR = trial.suggest_loguniform('learning_rate', 0.001, 0.01)
    LR = 0.001
    EPOSCHS = 200

    FEATURE = len(X_pos)
    dataset = Data_Split(x_pos=X_pos, y_pos=Y_pos, window_size=Window_size, predict_step=1, filter_size=gc.SSA_FILTER_COMPONENT(Y_pos),
                         feature=FEATURE)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataset.X_test_scaler = torch.from_numpy(dataset.X_test_scaler)

    waveNetlstm = WaveNet_LSTM(input_size=FEATURE, residual_size=RESIDUAL_SIZE, skip_size=SKIP_SIZE,
                        dilation_depth=DILATION_DEPTH, hidden_size=HIDDEN_SIZE)
    optimizer_lstm = torch.optim.Adam(waveNetlstm.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    # training and testing
    for epoch in range(EPOSCHS):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = torch.as_tensor(b_x, dtype=torch.float32)
            b_y = torch.as_tensor(b_y, dtype=torch.float32)

            out = waveNetlstm(b_x)
            loss = loss_func(out, b_y)
            optimizer_lstm.zero_grad()
            loss.backward()
            optimizer_lstm.step()
        if epoch % 20 == 0:
            print({"epoch": epoch, "loss": loss.item()})
    # lstm.to(device='cpu')
    dataset.X_test_scaler = torch.as_tensor(dataset.X_test_scaler, dtype=torch.float32)
    test_output = waveNetlstm(dataset.X_test_scaler)
    y_pred = test_output.detach().numpy()
    # 反归一化
    y_pred = dataset.scaler.inverse_transform(y_pred)
    y_true = dataset.y_test

    mae, mse, mape, smape, rmse, r2 = evaluate(y_true, y_pred)

    trial.report(mape, epoch)
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mape


if __name__ == "__main__":
    # 让结果可复现
    set_seed(1)
    sampler = TPESampler(seed=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # study = joblib.load('study_wavenet_lstm.pkl')

    results = pd.DataFrame(columns=['Indicator', 'MAPE'])
    for p in [0]:
        Y_pos = p
        X_pos = gc.CAUSAL_FACTORS(Y_pos)
        Window_size = gc.OPTIMAL_WINDOW_SIZE(Y_pos)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=100)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        # print("Study statistics: ")
        # print("  Number of finished trials: ", len(study.trials))
        # print("  Number of pruned trials: ", len(pruned_trials))
        # print("  Number of complete trials: ", len(complete_trials))

        trial = study.best_trial
        print(gc.INDICATORS()[Y_pos], " is done! ----Best trial Value: ", trial.value)
        results = results.append([{'Indicator': gc.INDICATORS()[Y_pos], 'MAPE': trial.value}])
        print("  Params: ")
        for key, value in trial.params.items():
            print("{}: {}".format(key, value))
        # 保存和恢复study
        joblib.dump(study, "study_wavenet_lstm_O3.pkl")
        # 恢复study
        # study= joblib.load('study.pkl')
    results.to_csv("HyperOptWaveNetLSTMResults.csv", encoding='utf-8')
