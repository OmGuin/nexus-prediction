import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from data import HOMAIRDataset
from utils import load_data
from model import TransformerRegression1, GFTNet, GFTNetX


CSV_PATH = Config.CSV_PATH
TARGET_COLUMN = "HOMA-IR"
BATCH_SIZE = Config.BATCH_SIZE
EPOCHS = Config.EPOCHS
LR = Config.LEARNING_RATE


X_train, X_val, y_train, y_val, scaler = load_data(CSV_PATH, TARGET_COLUMN)

train_dataset = HOMAIRDataset(X_train, y_train)
val_dataset = HOMAIRDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = Config.DEVICE
model = GFTNetX(input_dim=X_train.shape[1]
                              ).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)


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
            val_preds.extend(preds.cpu().numpy().flatten())
            val_targets.extend(y_batch.cpu().numpy().flatten())

    val_loss = mean_squared_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/len(train_loader):.4f}, Val MSE: {val_loss:.4f}, R²: {val_r2:.4f}")
