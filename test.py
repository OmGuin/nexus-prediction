import torch
from config import DEVICE
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def test(model, loader, criterion, plot=False):
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
    print(f"\nTest set: Average loss: {avg_loss:.6f}, MAE: {mae:.4f}, R^2: {r2:.6f}\n")
    if plot:
        y = y.cpu()
        y = np.array(y)  # Make sure y is a NumPy array
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_pred_denorm = np.array(all_preds) * y_std + y_mean
        y_test_denorm = np.array(all_targets) * y_std + y_mean

        plt.scatter(y_test_denorm, y_pred_denorm)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predictions vs True")
        min_val = min(min(y_test_denorm), min(y_pred_denorm))
        max_val = max(max(y_test_denorm), max(y_pred_denorm))

        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Diagonal line
        plt.show()
    return avg_loss, mae, r2
