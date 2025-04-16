import pandas as pd
import xgboost as xgb
from data import load_data
import pickle
from sklearn.metrics import accuracy_score
with open("params.pkl", "rb") as file:
  params = pickle.load(file)

X_train, X_test, y_train, y_test = load_data()

model = xgb.XGBRegressor(**params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_bin = (y_pred >= 0.5).astype(int)
print(accuracy_score(y_test, y_pred_bin))

with open("trained_xgb.pkl", "wb") as file:
  pickle.dump(model, file)
