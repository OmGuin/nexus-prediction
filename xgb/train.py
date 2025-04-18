import pandas as pd
import xgboost as xgb
from data import load_data
import pickle
with open("params.pkl", "rb") as file:
  params = pickle.load(file)

X_train, X_test, y_train, y_test = load_data()

#model = xgb.XGBRegressor(**params)

#X_train, X_test, y_train, y_test = load_data()
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#xgb_model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])



with open("trained_xgb.pkl", "wb") as file:
  pickle.dump(reg, file)
