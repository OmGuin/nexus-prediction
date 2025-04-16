import optuna
import xgboost as xgb
import pandas as pd
from data import load_data
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import cupy as cp

def objective(trial):
    params = {
        'device':'cuda',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),  # L2
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True)    # L1 
    }
    X_train, X_test, y_train, y_test = load_data()
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])
    
    y_pred = xgb_model.predict(dtest)
    mse = mean_squared_error(y_test, y_pred)

    return mse

def hp_tune():
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=100)
  
  with open('params.pkl', 'wb') as file:
    pickle.dump(study.best_trial.params, file)

  with open("study.pkl", "wb") as file:
    pickle.dump(study, file)

if __name__ == "__main__":
    hp_tune()
