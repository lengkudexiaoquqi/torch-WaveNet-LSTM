import optuna
from optuna.trial import TrialState
import joblib
study= joblib.load('study_bpnn_O3_降水量_湿度__params_hidden_size.pkl')
print(help(study))
print(study.trials)