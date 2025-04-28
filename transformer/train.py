import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from data import HOMAIRDataset
from utils import load_data
from model import TransformerRegression1, GFTNet, GFTNetX, SmallTabTransformer
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = Config.CSV_PATH
TARGET_COLUMN = "HOMA-IR"
BATCH_SIZE = Config.BATCH_SIZE
EPOCHS = Config.EPOCHS
LR = Config.LEARNING_RATE


X_train, X_val, y_train, y_val = load_data(CSV_PATH, TARGET_COLUMN)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

train_dataset = HOMAIRDataset(X_train, y_train)
val_dataset = HOMAIRDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# Add this right after loading data
print("Sample X values:", X_train[:3])
print("Sample y values:", y_train[:3])
print("Data types - X:", X_train.dtype, "y:", y_train.dtype)
device = Config.DEVICE
model = SmallTabTransformer(input_dim=X_train.shape[1],
                            d_model=32,  # Reduced from original
                            nhead=2,
                            num_layers=2,
                            dropout=0.2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=Config.WEIGHT_DECAY)
criterion = nn.HuberLoss()
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)


from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.9,
    reg_alpha=0.5
)
xgb.fit(X_train, y_train.ravel())
xgb_preds = xgb.predict(X_val)
xgb_mse = mean_squared_error(y_val, xgb_preds)
print("XGBoost R²:", r2_score(y_val, xgb_preds))
print("XGBoost MSE:", xgb_mse)
print("XGBoost Predictions:", xgb_preds[:5])
print("XGBoost True Values:", y_val[:5].flatten())
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())
            

    val_loss = criterion(torch.tensor(val_targets), torch.tensor(val_preds))
    val_r2 = r2_score(val_targets, val_preds)
    if(epoch == EPOCHS - 1 or epoch % 10 == 0):
        y = y_batch.cpu()
        y = np.array(y)  # Make sure y is a NumPy array
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_pred_denorm = np.array(val_preds) * y_std + y_mean
        y_test_denorm = np.array(val_targets) * y_std + y_mean

        plt.scatter(y_test_denorm, y_pred_denorm)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predictions vs True")
        min_val = min(min(y_test_denorm), min(y_pred_denorm))
        max_val = max(max(y_test_denorm), max(y_pred_denorm))

        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Diagonal line
        plt.show()
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/len(train_loader):.4f}, Val MSE: {val_loss:.4f}, R²: {val_r2:.4f}")
