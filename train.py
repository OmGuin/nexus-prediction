import pandas as pd
import xgboost as xgb
from data import load_data
import pickle

with open("params.pkl") as file:
  params = pickle.load(file)

X_train, X_test, y_train, y_test = load_data()

model = xgb.XGBRegressor(**params, use_label_encoder = False)

model.fit(X_train, y_train)

with open("trained_xgb.pkl", "wb") as file:
  pickle.dump(model, file)
