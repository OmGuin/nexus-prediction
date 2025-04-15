import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

data = pd.read_csv('.csv')

X = data.drop(columns=['IRScore'])
y = data['IRScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)

model.fit(X_train, y_train)

with open("trained_xgb.pkl", "wb") as file:
  pickle.dump(model, file)
