import optuna
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from data import load_data  # Your custom function that returns X_train, X_test, y_train, y_test

def objective(trial):
    # Define the hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Initialize model
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

    # Evaluate using cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

    # Return the average MSE (make it positive because we minimize)
    return -scores.mean()

def hp_tune():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Save best parameters
    with open('rf_regressor_params.pkl', 'wb') as f:
        pickle.dump(study.best_trial.params, f)

    # Save full study object
    with open("rf_regressor_study.pkl", "wb") as f:
        pickle.dump(study, f)

if __name__ == "__main__":
    hp_tune()
