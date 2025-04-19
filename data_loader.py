import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import CSV_PATH, BATCH_SIZE

class IRDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        
        self.scaler_X = StandardScaler()
        self.X_scaled = self.scaler_X.fit_transform(data.iloc[:, :-1].values).astype('float32')

        self.X = torch.tensor(self.X_scaled, dtype=torch.float32)
        self.Y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_loaders():
    full_dataset = IRDataset(CSV_PATH)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)
    return train_loader, test_loader