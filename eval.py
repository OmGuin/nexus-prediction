from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pickle
from data import load_data

_, _, _, y_test = load_data()

with open("trained_xgb.pkl", "rb") as file:
  model = pickle.dump(file)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

mape = mean_absolute_percentage_error(y_test, y_pred) * 100


print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
print(f"MAPE: {mape:.3f}%")
print(f"RMSE: {rmse:.3f}")
