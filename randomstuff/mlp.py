from sklearn.neural_network import MLPRegressor
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df= pd.read_csv(r'transformer\stuff.csv')

X = df.drop(columns=['HOMA-IR'])
X_train, X_test, y_train, y_test = train_test_split(X, df['HOMA-IR'], test_size=0.2, random_state=42)

# Neural Network
model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MLP - MSE: {mse:.3f}, R²: {r2:.3f}')
