import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from data import HOMAIRDataset
from utils import load_data
from model import TransformerRegression1, GFTNet, GFTNetX, SmallTabTransformer, BetterGFT
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = Config.CSV_PATH
TARGET_COLUMN = "HOMA-IR"
BATCH_SIZE = Config.BATCH_SIZE
EPOCHS = Config.EPOCHS
LR = Config.LEARNING_RATE


X_train, X_val, y_train, y_val = load_data(CSV_PATH, TARGET_COLUMN)


train_dataset = HOMAIRDataset(X_train, y_train)
val_dataset = HOMAIRDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# Add this right after loading data
device = Config.DEVICE

model = BetterGFT(input_dim=X_train.shape[1],
                            d_model=128,  # Reduced from original
                            nhead=8,
                            num_layers=4,
                            dropout=0.2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=Config.WEIGHT_DECAY)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)


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
    

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())
            
    val_targets = torch.tensor(val_targets, dtype=torch.float32)
    val_preds = torch.tensor(val_preds, dtype=torch.float32)
    val_loss = criterion(val_preds, val_targets)
    val_r2 = r2_score(val_targets.numpy(), val_preds.numpy())
    scheduler.step()



    # if(epoch == EPOCHS - 1 or epoch % 10 == 0):
    #     y = y_batch.cpu()
    #     y = np.array(y)  # Make sure y is a NumPy array
    #     y_mean = np.mean(y)
    #     y_std = np.std(y)
    #     y_pred_denorm = np.array(val_preds) * y_std + y_mean
    #     y_test_denorm = np.array(val_targets) * y_std + y_mean

    #     plt.scatter(y_test_denorm, y_pred_denorm)
    #     plt.xlabel("True Values")
    #     plt.ylabel("Predicted Values")
    #     plt.title("Predictions vs True")
    #     min_val = min(min(y_test_denorm), min(y_pred_denorm))
    #     max_val = max(max(y_test_denorm), max(y_pred_denorm))

    #     plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Diagonal line
    #     plt.show()
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/len(train_loader):.4f}, Val MSE: {val_loss:.4f}, R²: {val_r2:.4f}")
