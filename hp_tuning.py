import optuna
import xgboost as xgb
import pandas as pd
from data import load_data

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),  # L2
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True)    # L1 
    }
    X_train, X_test, y_train, y_test = load_data()
    xgb_model = xgb.XGBRegressor(**params, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)

    scores = cross_val_score(xgb_model, X_test, y_test, cv=5, scoring='accuracy')

    return np.mean(scores)

def hp_tune():
  
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=100)
  
  with open('params.pkl', 'wb') as file:
    pickle.dump(trial.params, file)


if __name__ == "__main__":
    hp_tune()
