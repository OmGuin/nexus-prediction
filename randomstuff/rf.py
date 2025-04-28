from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# Use original 5 features or PCA-transformed ones'

df= pd.read_csv(r'transformer\stuff.csv')
X = df.drop(columns=['HOMA-IR'])
X_train, X_test, y_train, y_test = train_test_split(X, df['HOMA-IR'], test_size=0.2, random_state=42)

# Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.3f}')
print(f'R² Score: {r2:.3f}')
