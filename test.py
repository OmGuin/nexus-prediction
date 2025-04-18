import torch
from config import DEVICE
from sklearn.metrics import mean_absolute_error, r2_score

def test(model, loader, criterion):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    print(f"\nTest set: Average loss: {avg_loss:.6f}, MAE: {mae:.4f}, R^2: {r2:.4f}\n")
    return avg_loss, mae, r2
