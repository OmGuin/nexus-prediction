from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from sklearn.metrics import mean_squared_error
from optuna.visualization import plot_optimization_history

df= pd.read_csv(r'transformer\stuff.csv')
df["Weight/Height"] = df["Body weight "] / df["Height "]
df["BMIxage"] = df["BMI"] * df["Age"]
df["Height^2"] = df["Height "] ** 2
X = df.drop(columns=['HOMA-IR'])
y = df['HOMA-IR']

X_train, X_test, y_train, y_test = train_test_split(X, df['HOMA-IR'], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def objective_rf(trial):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        max_depth=trial.suggest_int('max_depth', 2, 20),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

def objective_xgb(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        max_depth=trial.suggest_int('max_depth', 2, 15),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)

def objective_mlp(trial):
    model = MLPRegressor(
        hidden_layer_sizes=(
            trial.suggest_int('layer1', 32, 128),
            trial.suggest_int('layer2', 16, 64)
        ),
        activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
        learning_rate_init=trial.suggest_float('learning_rate_init', 0.0001, 0.01),
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    return mean_squared_error(y_test, preds)

def objective_svr(trial):
    model = SVR(
        C=trial.suggest_float('C', 0.1, 10.0),
        epsilon=trial.suggest_float('epsilon', 0.01, 0.5),
        kernel=trial.suggest_categorical('kernel', ['linear', 'rbf']),
    )
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    return mean_squared_error(y_test, preds)

# Running the studies
study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=50)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=50)

study_mlp = optuna.create_study(direction='minimize')
study_mlp.optimize(objective_mlp, n_trials=50)

study_svr = optuna.create_study(direction='minimize')
study_svr.optimize(objective_svr, n_trials=50)
plot_optimization_history(study_rf)
plot_optimization_history(study_xgb)
plot_optimization_history(study_mlp)
plot_optimization_history(study_mlp)


# Best parameters
print("Random Forest best params:", study_rf.best_params)
print("XGBoost best params:", study_xgb.best_params)
print("MLP best params:", study_mlp.best_params)
print("SVR best params:", study_svr.best_params)

# You can also check best MSEs:
print("Best RF MSE:", study_rf.best_value)
print("Best XGB MSE:", study_xgb.best_value)
print("Best MLP MSE:", study_mlp.best_value)
print("Best SVR MSE:", study_svr.best_value)