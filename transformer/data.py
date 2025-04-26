import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from config import Config
import numpy as np


class HOMAIRDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.log1p(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(y, dtype=torch.float32).flatten()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_loaders():
    dataset = HOMAIRDataset('subsample.csv')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle = False)

    return train_loader, test_loader


