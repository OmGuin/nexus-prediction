from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df= pd.read_csv(r'transformer\stuff.csv')

X = df.drop(columns=['HOMA-IR'])
X_train, X_test, y_train, y_test = train_test_split(X, df['HOMA-IR'], test_size=0.2, random_state=42)


# SVR needs scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR
model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'SVR - MSE: {mse:.3f}, R²: {r2:.3f}')
